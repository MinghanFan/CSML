"""
Advanced temporal round prediction model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from round_temporal_model import prepare_round_features, add_temporal_features


DATA_DIR = Path("clean_dataset")
OUTPUT_DIR = Path("clean_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = REGULATION_HALF_ROUNDS * 2
OVERTIME_HALF_ROUNDS = 3


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return (num / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def safe_log_loss(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> float:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return log_loss(y_true, proba, labels=[0, 1])


def load_base_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    round_players_df = pd.read_csv(data_dir / "round_players.csv")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    return rounds_df, round_players_df, matches_df


def build_team_round_stats(round_players_df: pd.DataFrame) -> pd.DataFrame:
    rp = round_players_df.copy()
    rp["survived"] = (
        rp["survived"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
    )

    agg = (
        rp.groupby(["match_id", "round_num", "team"])
        .agg(
            kills=("kills", "sum"),
            deaths=("deaths", "sum"),
            assists=("assists", "sum"),
            damage=("damage", "sum"),
            headshots=("headshots", "sum"),
            survivors=("survived", "sum"),
            equipment_value=("equipment_value", "mean"),
            cash_spent=("cash_spent", "mean"),
            money_start=("money_start", "mean"),
            money_end=("money_end", "mean"),
        )
        .reset_index()
    )

    wide = agg.pivot(index=["match_id", "round_num"], columns="team")
    wide.columns = [f"{metric}_{team}" for metric, team in wide.columns]
    wide = wide.reset_index()
    return wide


def compute_round_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "round_num"]).reset_index(drop=True)

    df["bomb_planted"] = (
        df["bomb_planted"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
    )

    df["round_duration"] = pd.to_numeric(df["round_duration"], errors="coerce").fillna(0.0)

    df["ct_win"] = (df["round_winner"] == "ct").astype(int)
    df["t_win"] = 1 - df["ct_win"]

    df["ct_score"] = df.groupby("match_id")["ct_win"].cumsum().shift(1).fillna(0).astype(int)
    df["t_score"] = df.groupby("match_id")["t_win"].cumsum().shift(1).fillna(0).astype(int)
    df["score_diff"] = df["ct_score"] - df["t_score"]
    df["score_total"] = df["ct_score"] + df["t_score"]
    df["ct_score_pct"] = safe_divide(df["ct_score"], df["score_total"].clip(lower=1))
    df["round_index"] = df["round_num"]

    df["is_first_half"] = (df["round_num"] <= REGULATION_HALF_ROUNDS).astype(int)
    df["is_second_half"] = (
        (df["round_num"] > REGULATION_HALF_ROUNDS)
        & (df["round_num"] <= REGULATION_TOTAL_ROUNDS)
    ).astype(int)
    df["is_overtime"] = (df["round_num"] > REGULATION_TOTAL_ROUNDS).astype(int)
    df["round_in_half"] = np.where(
        df["round_num"] <= REGULATION_HALF_ROUNDS,
        df["round_num"],
        np.where(
            df["round_num"] <= REGULATION_TOTAL_ROUNDS,
            df["round_num"] - REGULATION_HALF_ROUNDS,
            ((df["round_num"] - REGULATION_TOTAL_ROUNDS - 1) % OVERTIME_HALF_ROUNDS) + 1,
        ),
    )

    round_index = df.groupby("match_id").cumcount()
    df["round_progress"] = round_index / (round_index + 1).replace(0, 1)

    df["ct_win_streak"] = 0
    df["t_win_streak"] = 0

    for match_id, match_df in df.groupby("match_id"):
        ct_streak = []
        val = 0
        for won in match_df["ct_win"]:
            if won:
                val = val + 1 if val >= 0 else 1
            else:
                val = val - 1 if val <= 0 else -1
            ct_streak.append(val)
        df.loc[match_df.index, "ct_win_streak"] = pd.Series(ct_streak).shift(1).fillna(0)

        t_streak = []
        val = 0
        for won in match_df["t_win"]:
            if won:
                val = val + 1 if val >= 0 else 1
            else:
                val = val - 1 if val <= 0 else -1
            t_streak.append(val)
        df.loc[match_df.index, "t_win_streak"] = pd.Series(t_streak).shift(1).fillna(0)

    df["ct_win_streak_abs"] = df["ct_win_streak"].abs()
    df["momentum_index"] = df["score_diff"] + 0.5 * df["ct_win_streak"]
    df["pressure_index"] = df["score_diff"].abs() * df["round_progress"]

    df["round_end_reason_prev"] = (
        df.groupby("match_id")["round_end_reason"].shift(1).fillna("unknown")
    )

    reason_dummies = pd.get_dummies(df["round_end_reason_prev"], prefix="reason_prev")
    df = pd.concat([df, reason_dummies], axis=1)

    return df


def add_difference_features(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    for metric in metrics:
        ct_col = f"{metric}_ct"
        t_col = f"{metric}_t"
        if ct_col in df.columns and t_col in df.columns:
            df[f"{metric}_diff"] = df[ct_col] - df[t_col]
            df[f"{metric}_ratio"] = safe_divide(df[ct_col], df[t_col].replace(0, np.nan))
    return df


def add_temporal_aggregates(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    lag_rounds: Iterable[int] = (1, 2, 3, 5),
    rolling_windows: Iterable[int] = (3, 5),
    protected_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = df.sort_values(["match_id", "round_num"])

    protected = set(protected_cols or [])
    protected.update({"ct_equipment_value", "t_equipment_value"})

    for col in feature_cols:
        if col not in df.columns:
            continue

        group_series = df.groupby("match_id")[col]

        for lag in lag_rounds:
            df[f"{col}_lag{lag}"] = group_series.shift(lag)

        for window in rolling_windows:
            df[f"{col}_roll{window}"] = (
                group_series.shift(1).rolling(window=window, min_periods=1).mean()
            )

    drop_cols = [col for col in feature_cols if col not in protected and col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def build_advanced_dataset(data_dir: Path) -> pd.DataFrame:
    rounds_df, round_players_df, matches_df = load_base_tables(data_dir)
    team_stats = build_team_round_stats(round_players_df)

    df = rounds_df.merge(team_stats, on=["match_id", "round_num"], how="left")
    df = df.merge(matches_df[["match_id", "map_name"]], on="match_id", how="left")

    df = compute_round_context(df)

    df["equipment_diff"] = df["ct_equipment_value"] - df["t_equipment_value"]
    df["equipment_ratio"] = safe_divide(
        df["ct_equipment_value"], df["t_equipment_value"].replace(0, np.nan)
    )

    metrics = [
        "kills",
        "assists",
        "damage",
        "headshots",
        "survivors",
        "equipment_value",
        "cash_spent",
        "money_start",
        "money_end",
    ]
    df = add_difference_features(df, metrics)

    lag_candidates = [
        "kills_diff",
        "assists_diff",
        "damage_diff",
        "headshots_diff",
        "survivors_diff",
        "equipment_value_diff",
        "cash_spent_diff",
        "money_start_diff",
        "money_end_diff",
        "ct_win",
        "t_win",
        "round_duration",
        "bomb_planted",
    ]
    lag_candidates = [col for col in lag_candidates if col in df.columns]
    df = add_temporal_aggregates(
        df,
        lag_candidates,
        protected_cols={"ct_win", "t_win"},
    )

    map_dummies = pd.get_dummies(df["map_name"].fillna("unknown"), prefix="map")
    df = pd.concat([df, map_dummies], axis=1)

    safe_base_cols = [
        "match_id",
        "round_num",
        "round_winner",
        "ct_win",
        "t_win",
        "ct_score",
        "t_score",
        "score_diff",
        "score_total",
        "ct_score_pct",
        "round_index",
        "round_in_half",
        "round_progress",
        "is_first_half",
        "is_second_half",
        "is_overtime",
        "ct_win_streak",
        "t_win_streak",
        "ct_win_streak_abs",
        "momentum_index",
        "pressure_index",
    ]

    reason_cols = [c for c in df.columns if c.startswith("reason_prev_")]
    map_cols = [c for c in df.columns if c.startswith("map_")]
    temporal_cols = [
        c for c in df.columns if any(token in c for token in ("_lag", "_roll"))
    ]

    keep_cols = [
        col
        for col in safe_base_cols + reason_cols + map_cols + temporal_cols
        if col in df.columns
    ]

    df = df[keep_cols]
    df = df[df["round_in_half"] >= 3].reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


@dataclass
class CVSummary:
    params: Dict[str, float]
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    oof_predictions: np.ndarray
    best_iteration: int


def run_group_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: Dict[str, float],
    base_params: Dict[str, float],
    n_splits: int,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 75,
) -> CVSummary:
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred = np.zeros(len(X))
    records: List[Dict[str, float]] = []
    feature_importance = np.zeros(X.shape[1])
    best_iterations: List[int] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data)

        model = lgb.train(
            {**base_params, **params},
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
        )

        best_iter = model.best_iteration or num_boost_round
        best_iterations.append(best_iter)

        pred = model.predict(X.iloc[val_idx], num_iteration=best_iter)
        oof_pred[val_idx] = pred

        record = {
            "fold": fold,
            "auc": roc_auc_score(y.iloc[val_idx], pred),
            "logloss": safe_log_loss(y.iloc[val_idx], pred),
            "brier": brier_score_loss(y.iloc[val_idx], pred),
            "accuracy": accuracy_score(y.iloc[val_idx], (pred >= 0.5).astype(int)),
        }
        records.append(record)
        feature_importance += model.feature_importance(importance_type="gain")

    metrics_df = pd.DataFrame(records)
    feature_df = pd.DataFrame(
        {"feature": X.columns, "importance": feature_importance / n_splits}
    ).sort_values("importance", ascending=False)
    avg_best_iter = int(np.mean(best_iterations)) if best_iterations else num_boost_round

    return CVSummary(
        params=params,
        metrics=metrics_df,
        feature_importance=feature_df,
        oof_predictions=oof_pred,
        best_iteration=avg_best_iter,
    )


def evaluate_baseline(
    rounds_df: pd.DataFrame,
    round_players_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    min_round: int = 3,
) -> Dict[str, float]:
    base_df = prepare_round_features(rounds_df, round_players_df, matches_df)
    base_df = add_temporal_features(base_df)

    feature_cols = [
        col for col in base_df.columns if col not in {"match_id", "round_num", "round_winner", "ct_won", "map_name"}
    ]

    mask = base_df["round_in_half"] >= min_round
    X = base_df.loc[mask, feature_cols].fillna(0).reset_index(drop=True)
    y = base_df.loc[mask, "round_winner"].astype(int).reset_index(drop=True)
    groups = base_df.loc[mask, "match_id"].reset_index(drop=True)
    maps = base_df.loc[mask, "map_name"].fillna("unknown").reset_index(drop=True)

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Baseline evaluation requires at least two matches.")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbosity": -1,
    }

    gkf = GroupKFold(n_splits=n_splits)
    records = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        train_maps = maps.iloc[train_idx].reset_index(drop=True)
        train_targets = y.iloc[train_idx].reset_index(drop=True)
        train_map_df = pd.DataFrame({"map": train_maps, "ct_win": train_targets})
        map_ct_rate = train_map_df.groupby("map")["ct_win"].mean()
        eps = 1e-6
        map_odds_ratio = ((map_ct_rate + eps) / (1 - map_ct_rate + eps)).to_dict()
        fallback_ratio = float(np.mean(list(map_odds_ratio.values())) if map_odds_ratio else 1.0)

        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=400,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_iteration = model.best_iteration or 400
        pred = model.predict(X.iloc[val_idx], num_iteration=best_iteration)

        val_maps = maps.iloc[val_idx]
        map_ratio = val_maps.map(map_odds_ratio).fillna(fallback_ratio).to_numpy()
        odds = pred / np.clip(1 - pred, 1e-6, None)
        adjusted_probs = odds * map_ratio
        adjusted_probs = adjusted_probs / (1 + adjusted_probs)
        adjusted_probs = np.clip(adjusted_probs, 1e-6, 1 - 1e-6)

        records.append(
            {
                "fold": fold,
                "auc": roc_auc_score(y.iloc[val_idx], adjusted_probs),
                "logloss": safe_log_loss(y.iloc[val_idx], adjusted_probs),
                "brier": brier_score_loss(y.iloc[val_idx], adjusted_probs),
                "accuracy": accuracy_score(y.iloc[val_idx], (adjusted_probs >= 0.5).astype(int)),
            }
        )

    metrics_df = pd.DataFrame(records)
    return {
        "auc": metrics_df["auc"].mean(),
        "accuracy": metrics_df["accuracy"].mean(),
        "logloss": metrics_df["logloss"].mean(),
        "brier": metrics_df["brier"].mean(),
        "fold_metrics": metrics_df,
    }


def train_advanced_model() -> Dict[str, object]:
    advanced_df = build_advanced_dataset(DATA_DIR)

    target = advanced_df["ct_win"].astype(int)
    groups = advanced_df["match_id"]

    drop_cols = {"match_id", "round_num", "round_winner", "ct_win", "t_win", "round_end_reason", "round_end_reason_prev", "map_name"}
    feature_cols = [
        col for col in advanced_df.columns if col not in drop_cols and advanced_df[col].dtype != "O"
    ]
    X = advanced_df[feature_cols].fillna(0)

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Advanced model requires at least two unique matches.")

    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "min_data_in_bin": 16,
        "max_bin": 255,
        "seed": 42,
    }

    candidate_params: Iterable[Dict[str, float]] = [
        {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.85,
            "bagging_freq": 4,
            "min_child_samples": 20,
            "lambda_l1": 0.0,
            "lambda_l2": 0.1,
        },
        {
            "num_leaves": 95,
            "learning_rate": 0.035,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 30,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        },
        {
            "num_leaves": 127,
            "learning_rate": 0.03,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.75,
            "bagging_freq": 2,
            "min_child_samples": 35,
            "lambda_l1": 0.2,
            "lambda_l2": 0.0,
        },
    ]

    cv_summaries: List[CVSummary] = []
    for params in candidate_params:
        cv_summary = run_group_kfold(
            X,
            target,
            groups,
            params=params,
            base_params=base_params,
            n_splits=n_splits,
        )
        cv_summaries.append(cv_summary)

    best_summary = max(cv_summaries, key=lambda s: s.metrics["auc"].mean())

    # Calibrate using out-of-fold predictions from CV models.
    # Note: this calibration is trained on folds' OOF predictions, so applying it
    # to the final full-data model may be slightly optimistic versus a true hold-out set.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(best_summary.oof_predictions, target)
    calibrated = calibrator.predict(best_summary.oof_predictions)
    calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)

    advanced_metrics = {
        "auc": roc_auc_score(target, calibrated),
        "accuracy": accuracy_score(target, (calibrated >= 0.5).astype(int)),
        "logloss": safe_log_loss(target, calibrated),
        "brier": brier_score_loss(target, calibrated),
    }

    final_params = {**base_params, **best_summary.params}
    final_model = lgb.train(
        final_params,
        lgb.Dataset(X, label=target),
        num_boost_round=best_summary.best_iteration,
    )

    model_path = OUTPUT_DIR / "advanced_temporal_model.lgb.txt"
    final_model.save_model(str(model_path))
    joblib.dump(
        {"calibrator": calibrator, "feature_columns": feature_cols},
        OUTPUT_DIR / "advanced_temporal_model_calibrator.pkl",
    )

    metrics_path = OUTPUT_DIR / "advanced_temporal_cv_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "advanced_model": {
                    **advanced_metrics,
                    "best_params": best_summary.params,
                    "best_iteration": best_summary.best_iteration,
                }
            },
            handle,
            indent=2,
        )

    best_summary.feature_importance.to_csv(
        OUTPUT_DIR / "advanced_temporal_feature_importance.csv", index=False
    )
    best_summary.metrics.to_csv(
        OUTPUT_DIR / "advanced_temporal_cv_results.csv", index=False
    )

    return {
        "metrics": advanced_metrics,
        "feature_importance": best_summary.feature_importance,
        "cv_metrics": best_summary.metrics,
    }


def main() -> None:
    print("=" * 80)
    print("ADVANCED TEMPORAL ROUND PREDICTION (NO EVENT PARQUETS REQUIRED)")
    print("=" * 80)

    if not DATA_DIR.exists():
        raise FileNotFoundError("clean_dataset directory not found. Run the data pipeline first.")

    rounds_df, round_players_df, matches_df = load_base_tables(DATA_DIR)

    print("\nEvaluating baseline (round_temporal_model.py equivalent)...")
    baseline = evaluate_baseline(rounds_df, round_players_df, matches_df)
    print(
        f"Baseline -> AUC: {baseline['auc']:.4f}, "
        f"Accuracy: {baseline['accuracy']:.4f}, "
        f"LogLoss: {baseline['logloss']:.4f}, "
        f"Brier: {baseline['brier']:.4f}"
    )

    print("\nTraining advanced model...")
    advanced = train_advanced_model()
    metrics = advanced["metrics"]
    print(
        f"Advanced -> AUC: {metrics['auc']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"LogLoss: {metrics['logloss']:.4f}, "
        f"Brier: {metrics['brier']:.4f}"
    )

    deltas = {
        "auc": metrics["auc"] - baseline["auc"],
        "accuracy": metrics["accuracy"] - baseline["accuracy"],
        "logloss": metrics["logloss"] - baseline["logloss"],
        "brier": metrics["brier"] - baseline["brier"],
    }
    print(
        f"\nImprovements -> ΔAUC: {deltas['auc']:+.4f}, "
        f"ΔAccuracy: {deltas['accuracy']:+.4f}, "
        f"ΔLogLoss: {deltas['logloss']:+.4f}, "
        f"ΔBrier: {deltas['brier']:+.4f}"
    )

    comparison = {
        "baseline": {
            "auc": baseline["auc"],
            "accuracy": baseline["accuracy"],
            "logloss": baseline["logloss"],
            "brier": baseline["brier"],
        },
        "advanced": metrics,
        "delta": deltas,
    }
    with (OUTPUT_DIR / "advanced_temporal_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    print("\nArtifacts written to clean_dataset/:")
    print("  - advanced_temporal_model.lgb.txt")
    print("  - advanced_temporal_model_calibrator.pkl")
    print("  - advanced_temporal_cv_metrics.json")
    print("  - advanced_temporal_cv_results.csv")
    print("  - advanced_temporal_feature_importance.csv")
    print("  - advanced_temporal_comparison.json")


if __name__ == "__main__":
    main()
