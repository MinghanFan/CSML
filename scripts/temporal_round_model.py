"""
Temporal Round Prediction Model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# Constants
DATA_DIR = Path("clean_dataset")
OUTPUT_DIR = Path("clean_dataset")
REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = 24
OVERTIME_HALF_ROUNDS = 3


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    result = num / denom.replace(0, np.nan)
    return result.fillna(0.0)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load base CSV files."""
    rounds = pd.read_csv(DATA_DIR / "rounds.csv")
    players = pd.read_csv(DATA_DIR / "round_players.csv")
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    return rounds, players, matches


def compute_team_stats(round_players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player stats by team for each round."""
    rp = round_players.copy()
    
    # Convert survived to numeric
    rp["survived"] = rp["survived"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
    
    # Aggregate by match, round, and team
    stats = (
        rp.groupby(["match_id", "round_num", "team"])
        .agg(
            kills=("kills", "sum"),
            deaths=("deaths", "sum"),
            assists=("assists", "sum"),
            damage=("damage", "sum"),
            headshots=("headshots", "sum"),
            survivors=("survived", "sum"),
        )
        .reset_index()
    )
    
    # Pivot to get CT and T columns
    wide = stats.pivot(index=["match_id", "round_num"], columns="team")
    wide.columns = [f"{stat}_{team}" for stat, team in wide.columns]
    return wide.reset_index()


def build_features(rounds: pd.DataFrame, players: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build feature set from raw data."""
    
    print("Building features...")
    
    # Get team stats
    team_stats = compute_team_stats(players)
    
    # Merge rounds with team stats and map info
    df = rounds.merge(team_stats, on=["match_id", "round_num"], how="left")
    df = df.merge(matches[["match_id", "map_name"]], on="match_id", how="left")
    
    # Sort by match and round
    df = df.sort_values(["match_id", "round_num"]).reset_index(drop=True)
    
    # ==========================================
    # TARGET VARIABLE
    # ==========================================
    df["ct_win"] = (df["round_winner"] == "ct").astype(int)
    
    # ==========================================
    # GAME STATE (available at round start)
    # ==========================================
    
    # Round context
    df["round_num_normalized"] = df["round_num"] / 30.0  # Normalize to [0, 1] range
    df["is_first_half"] = (df["round_num"] <= REGULATION_HALF_ROUNDS).astype(int)
    df["is_second_half"] = ((df["round_num"] > REGULATION_HALF_ROUNDS) & 
                             (df["round_num"] <= REGULATION_TOTAL_ROUNDS)).astype(int)
    df["is_overtime"] = (df["round_num"] > REGULATION_TOTAL_ROUNDS).astype(int)
    
    # Score tracking (cumulative before current round)
    df["ct_score"] = df.groupby("match_id")["ct_win"].cumsum().shift(1).fillna(0).astype(int)
    df["t_score"] = df.groupby("match_id", group_keys=False).apply(
        lambda x: (1 - x["ct_win"]).cumsum().shift(1).fillna(0)
    ).astype(int)
    
    df["score_diff"] = df["ct_score"] - df["t_score"]
    df["score_total"] = df["ct_score"] + df["t_score"]
    df["ct_score_pct"] = safe_divide(df["ct_score"], df["score_total"].clip(lower=1))
    
    # Win momentum (recent performance)
    df["ct_won_prev"] = df.groupby("match_id")["ct_win"].shift(1).fillna(0.5)
    
    # Win streaks
    for match_id, group in df.groupby("match_id"):
        streak = 0
        streaks = []
        for won in group["ct_win"]:
            if won:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
            streaks.append(streak)
        
        # Shift to get streak entering this round
        df.loc[group.index, "ct_win_streak"] = pd.Series(streaks).shift(1).fillna(0).values
    
    # ==========================================
    # HISTORICAL PERFORMANCE (lag features)
    # ==========================================
    
    # Compute differences from past rounds (these happened, so no leakage)
    df["kills_diff_actual"] = df["kills_ct"] - df["kills_t"]
    df["damage_diff_actual"] = df["damage_ct"] - df["damage_t"]
    df["survivors_diff_actual"] = df["survivors_ct"] - df["survivors_t"]
    df["headshot_pct_ct_actual"] = safe_divide(df["headshots_ct"], df["kills_ct"].clip(lower=1))
    df["headshot_pct_t_actual"] = safe_divide(df["headshots_t"], df["kills_t"].clip(lower=1))
    
    # Economy (from previous round)
    df["equipment_value_ct_prev"] = df.groupby("match_id")["ct_equipment_value"].shift(1)
    df["equipment_value_t_prev"] = df.groupby("match_id")["t_equipment_value"].shift(1)
    df["equipment_diff_prev"] = df["equipment_value_ct_prev"] - df["equipment_value_t_prev"]
    
    # Create lag features (shift by 1 to avoid leakage)
    lag_features = [
        "kills_diff_actual",
        "damage_diff_actual", 
        "survivors_diff_actual",
        "headshot_pct_ct_actual",
        "headshot_pct_t_actual",
    ]
    
    for feature in lag_features:
        # Individual lags
        for lag in [1, 2, 3]:
            df[f"{feature}_lag{lag}"] = df.groupby("match_id")[feature].shift(lag)
        
        # Rolling averages
        for window in [3, 5]:
            df[f"{feature}_roll{window}"] = (
                df.groupby("match_id")[feature]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    
    # ==========================================
    # MAP ENCODING
    # ==========================================
    map_dummies = pd.get_dummies(df["map_name"].fillna("unknown"), prefix="map")
    df = pd.concat([df, map_dummies], axis=1)
    
    # ==========================================
    # CLEAN UP
    # ==========================================
    
    # Drop actual round outcome columns (would leak target)
    leak_cols = [
        "round_winner", "round_end_reason", "bomb_planted", "bomb_site",
        "ct_players_alive_end", "t_players_alive_end",
        "ct_equipment_value", "t_equipment_value",
        "kills_ct", "kills_t", "deaths_ct", "deaths_t",
        "assists_ct", "assists_t", "damage_ct", "damage_t",
        "headshots_ct", "headshots_t", "survivors_ct", "survivors_t",
        "kills_diff_actual", "damage_diff_actual", "survivors_diff_actual",
        "headshot_pct_ct_actual", "headshot_pct_t_actual",
        "round_duration"
    ]
    
    df = df.drop(columns=[col for col in leak_cols if col in df.columns])
    
    # Drop any remaining object columns (except identifiers we'll exclude later)
    object_cols = df.select_dtypes(include=['object']).columns
    cols_to_drop = [col for col in object_cols if col not in ['match_id', 'map_name']]
    if cols_to_drop:
        print(f"Dropping object columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    print(f"Created dataset with {len(df)} rounds and {len(df.columns)} columns")
    return df


def train_model(df: pd.DataFrame, min_round: int = 3) -> Dict:
    """Train and evaluate the model with cross-validation."""
    
    print(f"\nTraining model (filtering rounds < {min_round})...")
    
    # Filter early rounds (insufficient history)
    df_train = df[df["round_num"] >= min_round].copy().reset_index(drop=True)
    
    # Separate features and target
    exclude = ["match_id", "round_num", "ct_win", "map_name"]
    feature_cols = [col for col in df_train.columns if col not in exclude]
    
    X = df_train[feature_cols]
    
    # Safety check: ensure no object columns in features
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"Warning: Dropping object columns from features: {object_cols}")
        X = X.drop(columns=object_cols)
        feature_cols = [col for col in feature_cols if col not in object_cols]
    
    y = df_train["ct_win"]
    groups = df_train["match_id"]
    maps = df_train["map_name"].fillna("unknown")
    
    print(f"Training set: {len(X)} rounds, {len(feature_cols)} features")
    print(f"CT win rate: {y.mean():.1%}")
    print(f"Number of matches: {groups.nunique()}")
    
    # Model parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": 42,
    }
    
    # Cross-validation
    n_splits = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    
    cv_scores = []
    feature_importance = np.zeros(len(feature_cols))
    oof_predictions = np.zeros(len(X))
    best_iterations = []
    
    print(f"\nRunning {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )
        
        best_iter = model.best_iteration or 500
        best_iterations.append(best_iter)
        
        # Predict
        pred = model.predict(X_val, num_iteration=best_iter)
        oof_predictions[val_idx] = pred
        
        # Evaluate
        pred_binary = (pred >= 0.5).astype(int)
        scores = {
            "fold": fold,
            "auc": roc_auc_score(y_val, pred),
            "accuracy": accuracy_score(y_val, pred_binary),
            "logloss": log_loss(y_val, np.clip(pred, 1e-7, 1 - 1e-7)),
            "brier": brier_score_loss(y_val, pred),
        }
        cv_scores.append(scores)
        
        print(
            f"  Fold {fold}: AUC={scores['auc']:.4f}, "
            f"Acc={scores['accuracy']:.3f}, LogLoss={scores['logloss']:.4f}"
        )
        
        # Accumulate feature importance
        feature_importance += model.feature_importance(importance_type="gain")
    
    # Average scores and importance
    cv_scores_df = pd.DataFrame(cv_scores)
    feature_importance /= n_splits
    
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"AUC:      {cv_scores_df['auc'].mean():.4f} ± {cv_scores_df['auc'].std():.4f}")
    print(f"Accuracy: {cv_scores_df['accuracy'].mean():.3f} ± {cv_scores_df['accuracy'].std():.3f}")
    print(f"LogLoss:  {cv_scores_df['logloss'].mean():.4f} ± {cv_scores_df['logloss'].std():.4f}")
    print(f"Brier:    {cv_scores_df['brier'].mean():.4f} ± {cv_scores_df['brier'].std():.4f}")
    
    # Calibrate predictions
    print("\nCalibrating model...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_predictions, y)
    calibrated_pred = calibrator.predict(oof_predictions)
    
    calibrated_scores = {
        "auc": roc_auc_score(y, calibrated_pred),
        "accuracy": accuracy_score(y, (calibrated_pred >= 0.5).astype(int)),
        "logloss": log_loss(y, np.clip(calibrated_pred, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y, calibrated_pred),
    }
    
    print(f"After calibration: AUC={calibrated_scores['auc']:.4f}, "
          f"Acc={calibrated_scores['accuracy']:.3f}")
    
    # Train final model on all data
    print("\nTraining final model on full dataset...")
    avg_best_iter = int(np.mean(best_iterations))
    
    final_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=avg_best_iter,
    )
    
    # Save model and artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUT_DIR / "round_prediction_model.txt"
    final_model.save_model(str(model_path))
    
    joblib.dump(
        {
            "calibrator": calibrator,
            "feature_columns": feature_cols,
            "min_round": min_round,
        },
        OUTPUT_DIR / "model_artifacts.pkl",
    )
    
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    cv_scores_df.to_csv(OUTPUT_DIR / "cv_scores.csv", index=False)
    
    metrics = {
        "cv_mean": cv_scores_df.mean().to_dict(),
        "cv_std": cv_scores_df.std().to_dict(),
        "calibrated": calibrated_scores,
        "best_iteration": avg_best_iter,
    }
    
    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return {
        "model": final_model,
        "calibrator": calibrator,
        "importance": importance_df,
        "cv_scores": cv_scores_df,
        "metrics": metrics,
    }


def analyze_features(importance_df: pd.DataFrame):
    """Analyze and print feature importance insights."""
    
    print(f"\n{'='*60}")
    print("TOP 15 MOST IMPORTANT FEATURES")
    print(f"{'='*60}")
    
    for i, row in importance_df.head(15).iterrows():
        print(f"{row['feature']:40s} {row['importance']:8.1f}")
    
    # Feature categories
    categories = {
        "Score/Momentum": importance_df[
            importance_df["feature"].str.contains("score|streak|won_prev")
        ]["importance"].sum(),
        "Historical Performance": importance_df[
            importance_df["feature"].str.contains("lag|roll")
        ]["importance"].sum(),
        "Round Context": importance_df[
            importance_df["feature"].str.contains("round_num|half|overtime")
        ]["importance"].sum(),
        "Economy": importance_df[
            importance_df["feature"].str.contains("equipment")
        ]["importance"].sum(),
        "Map": importance_df[
            importance_df["feature"].str.contains("map_")
        ]["importance"].sum(),
    }
    
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE BY CATEGORY")
    print(f"{'='*60}")
    
    total = sum(categories.values())
    for category, value in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (value / total * 100) if total > 0 else 0
        print(f"{category:30s} {pct:5.1f}%")


def main():
    """Main execution."""
    
    print("=" * 80)
    print("TEMPORAL ROUND PREDICTION MODEL")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    rounds, players, matches = load_data()
    print(f"Loaded {len(rounds)} rounds from {rounds['match_id'].nunique()} matches")
    
    # Build features
    df = build_features(rounds, players, matches)
    
    # Train model
    results = train_model(df, min_round=3)
    
    # Analyze features
    analyze_features(results["importance"])
    
    print(f"\n{'='*60}")
    print("SAVED FILES")
    print(f"{'='*60}")
    print("  ✓ round_prediction_model.txt")
    print("  ✓ model_artifacts.pkl")
    print("  ✓ feature_importance.csv")
    print("  ✓ cv_scores.csv")
    print("  ✓ model_metrics.json")
    
    print(f"\n{'='*60}")
    print("MODEL COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()