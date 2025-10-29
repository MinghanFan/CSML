"""
Temporal Round Prediction Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = REGULATION_HALF_ROUNDS * 2
OVERTIME_HALF_ROUNDS = 3
OVERTIME_BLOCK_ROUNDS = OVERTIME_HALF_ROUNDS * 2


def prepare_round_features(rounds_df, round_players_df, matches_df):
    """Prepare features from existing round data"""
    
    print("Preparing round features...")
    
    match_to_map = matches_df.set_index('match_id')['map_name'].to_dict()
    features_list = []
    
    for match_id in rounds_df['match_id'].unique():
        match_rounds = rounds_df[rounds_df['match_id'] == match_id].sort_values('round_num')
        match_round_players = round_players_df[round_players_df['match_id'] == match_id]
        
        for idx, round_row in match_rounds.iterrows():
            round_num = round_row['round_num']
            round_players = match_round_players[match_round_players['round_num'] == round_num]
            
            # Aggregate by team
            ct_players = round_players[round_players['team'] == 'ct']
            t_players = round_players[round_players['team'] == 't']
            
            # Create features
            features = {
                'match_id': match_id,
                'round_num': round_num,
                'round_winner': 1 if round_row['round_winner'] == 'ct' else 0,
                
                # Economy snapshot (used only for lag features later)
                'equipment_diff': round_row['ct_equipment_value'] - round_row['t_equipment_value'],
                'ct_equipment_value': round_row['ct_equipment_value'],
                't_equipment_value': round_row['t_equipment_value'],
                
                # Round outcome indicators (dropped before modeling, kept for history features)
                'alive_diff_end': round_row['ct_players_alive_end'] - round_row['t_players_alive_end'],
                'bomb_planted': 1 if round_row['bomb_planted'] else 0,
                
                # Performance metrics (used only for lag/rolling stats)
                'kill_diff': ct_players['kills'].sum() - t_players['kills'].sum(),
                'damage_diff': ct_players['damage'].sum() - t_players['damage'].sum(),
                'survivor_diff': ct_players['survived'].sum() - t_players['survived'].sum(),
                
                # Round context
                'round_in_half': (
                    round_num
                    if round_num <= REGULATION_HALF_ROUNDS
                    else (
                        round_num - REGULATION_HALF_ROUNDS
                        if round_num <= REGULATION_TOTAL_ROUNDS
                        else ((round_num - REGULATION_TOTAL_ROUNDS - 1) % OVERTIME_HALF_ROUNDS) + 1
                    )
                )
            }
            
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df['map_name'] = df['match_id'].map(match_to_map).fillna('unknown')
    return df


def add_temporal_features(df, lag_rounds=[1, 2, 3, 5], window_sizes=[3, 5]):
    """Add momentum and lag features"""
    
    print("Adding temporal features...")
    df = df.sort_values(['match_id', 'round_num'])
    
    # Calculate streaks
    df['ct_won'] = df['round_winner'].astype(int)
    df['t_won'] = 1 - df['ct_won']
    
    for match_id in df['match_id'].unique():
        match_mask = df['match_id'] == match_id
        match_df = df[match_mask].copy()
        
        # Win streak
        streak = 0
        streaks = []
        for won in match_df['ct_won']:
            if won:
                streak += 1
            else:
                streak = -1 if streak <= 0 else 0
            streaks.append(streak)
        
        df.loc[match_mask, 'ct_streak'] = streaks
    
    # Align streak context to the start of each round
    df['ct_streak'] = df.groupby('match_id')['ct_streak'].shift(1).fillna(0)
    
    # Cumulative score (state entering the round)
    df['ct_score'] = df.groupby('match_id')['ct_won'].cumsum().shift(1).fillna(0).astype(int)
    df['t_score'] = df.groupby('match_id')['t_won'].cumsum().shift(1).fillna(0).astype(int)
    df['score_diff'] = df['ct_score'] - df['t_score']
    
    # Lag features
    key_features = ['kill_diff', 'damage_diff', 'equipment_diff', 'ct_streak', 'score_diff']
    
    for col in key_features:
        if col in df.columns:
            for lag in lag_rounds:
                df[f'{col}_lag{lag}'] = df.groupby('match_id')[col].shift(lag)
    
    # Rolling averages
    for col in ['kill_diff', 'damage_diff']:
        if col in df.columns:
            for window in window_sizes:
                df[f'{col}_roll{window}'] = (
                    df.groupby('match_id')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                    .shift(1)  # Avoid leakage
                )
    
    # Drop columns that leak round-t outcomes
    leaky_cols = [
        'ct_won', 't_won',
        'equipment_diff', 'ct_equipment_value', 't_equipment_value',
        'alive_diff_end', 'bomb_planted',
        'kill_diff', 'damage_diff', 'survivor_diff'
    ]
    df = df.drop(columns=[col for col in leaky_cols if col in df.columns])
    
    return df


def train_temporal_model(df, map_odds_ratio, min_round=3):
    """Train LightGBM model with temporal features"""
    
    print("\nTraining temporal model...")
    
    # Filter out early rounds (no history)
    # Prepare features
    exclude_cols = ['match_id', 'round_num', 'round_winner', 'ct_won', 'map_name']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_full = df[feature_cols]
    y_full = df['round_winner'].astype(int)
    map_names_full = df['map_name'].fillna('unknown')
    match_ids_full = df['match_id']
    round_nums_full = df['round_num']

    baseline_rate = y_full.mean()
    baseline_accuracy = max(baseline_rate, 1 - baseline_rate)
    try:
        baseline_auc = roc_auc_score(y_full, np.full(len(y_full), baseline_rate, dtype=float))
    except ValueError:
        baseline_auc = float("nan")
    print(
        f"Baseline constant prediction -> "
        f"Accuracy={baseline_accuracy:.3f}, AUC={baseline_auc:.3f}"
    )

    train_mask = df['round_in_half'] >= min_round
    X = X_full[train_mask].reset_index(drop=True)
    y = y_full[train_mask].reset_index(drop=True)
    groups = match_ids_full[train_mask].reset_index(drop=True)
    map_names_train = map_names_full[train_mask].reset_index(drop=True)
    
    print(f"Dataset: {len(X)} rounds, {len(X.columns)} features")
    print(f"CT win rate: {y.mean():.2%}")
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbosity': -1
    }
    
    # Group K-fold CV
    gkf = GroupKFold(n_splits=5)
    scores = []
    feature_importance = np.zeros(len(feature_cols))
    
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        best_iterations.append(model.best_iteration or 300)
        
        # Evaluate
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        val_maps = map_names_train.iloc[val_idx].values
        ratios = np.array([map_odds_ratio.get(m, 1.0) for m in val_maps])
        odds = val_pred / np.clip(1 - val_pred, 1e-9, None)
        adjusted_odds = odds * ratios
        val_pred = adjusted_odds / (1 + adjusted_odds)
        val_pred = np.clip(val_pred, 1e-6, 1 - 1e-6)
        val_pred_labels = (val_pred >= 0.5).astype(int)
        
        fold_scores = {
            'fold': fold,
            'logloss': log_loss(y_val, val_pred),
            'auc': roc_auc_score(y_val, val_pred),
            'brier': brier_score_loss(y_val, val_pred),
            'accuracy': accuracy_score(y_val, val_pred_labels),
        }
        scores.append(fold_scores)
        
        # Accumulate feature importance
        feature_importance += model.feature_importance(importance_type='gain')
        
        print(
            f"Fold {fold}: "
            f"LogLoss={fold_scores['logloss']:.4f}, "
            f"AUC={fold_scores['auc']:.4f}, "
            f"Brier={fold_scores['brier']:.4f}, "
            f"Accuracy={fold_scores['accuracy']:.3f}"
        )
    
    # Average feature importance
    feature_importance /= 5
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Summary
    scores_df = pd.DataFrame(scores)
    print(
        f"\nMean scores: "
        f"LogLoss={scores_df['logloss'].mean():.4f}, "
        f"AUC={scores_df['auc'].mean():.4f}, "
        f"Brier={scores_df['brier'].mean():.4f}, "
        f"Accuracy={scores_df['accuracy'].mean():.3f}"
    )

    # Train final model on full dataset for inference
    final_num_boost_round = int(np.mean(best_iterations)) if best_iterations else 300
    X_train = X.fillna(0)
    X_full_filled = X_full.fillna(0)
    
    final_train_data = lgb.Dataset(X_train, label=y)
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=final_num_boost_round,
    )
    final_model.save_model("temporal_model.lgb.txt")
    full_probs = final_model.predict(X_full_filled, num_iteration=final_num_boost_round)
    map_ratios_full = map_names_full.map(map_odds_ratio).fillna(1.0).values
    full_odds = full_probs / np.clip(1 - full_probs, 1e-6, None)
    adjusted_full_odds = full_odds * map_ratios_full
    full_probs = adjusted_full_odds / (1 + adjusted_full_odds)
    preds_df = pd.DataFrame({
        'match_id': match_ids_full.values,
        'round_num': round_nums_full.values,
        'map_name': map_names_full.values,
        'round_winner': y_full.values,
        'ct_win_prob': full_probs,
    })
    preds_df.to_csv('temporal_predictions.csv', index=False)
    
    return final_model, importance_df, scores_df


def plot_results(importance_df, scores_df):
    """Plot feature importance and model performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance
    top_features = importance_df.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Feature Importances')
    axes[0, 0].invert_yaxis()
    
    # CV scores
    scores_melted = scores_df.melt(id_vars='fold', var_name='metric', value_name='score')
    scores_melted = scores_melted[scores_melted['metric'] != 'fold']
    
    sns.boxplot(data=scores_melted, x='metric', y='score', ax=axes[0, 1])
    axes[0, 1].set_title('Cross-Validation Scores')
    axes[0, 1].set_xlabel('Metric')
    axes[0, 1].set_ylabel('Score')
    
    # Temporal feature importance
    temporal_features = importance_df[
        importance_df['feature'].str.contains('lag|roll|streak|score_diff|cumsum')
    ].head(10)
    
    axes[1, 0].barh(range(len(temporal_features)), temporal_features['importance'])
    axes[1, 0].set_yticks(range(len(temporal_features)))
    axes[1, 0].set_yticklabels(temporal_features['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top Temporal Features')
    axes[1, 0].invert_yaxis()
    
    # Feature categories
    feature_categories = {
        'Equipment': importance_df[importance_df['feature'].str.contains('equipment|equip')]['importance'].sum(),
        'Performance': importance_df[importance_df['feature'].str.contains('kill|damage|survivor')]['importance'].sum(),
        'Temporal': importance_df[importance_df['feature'].str.contains('lag|roll|streak|cumsum')]['importance'].sum(),
        'Context': importance_df[importance_df['feature'].str.contains('round|half|bomb|alive')]['importance'].sum(),
    }
    
    axes[1, 1].pie(feature_categories.values(), labels=feature_categories.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Feature Importance by Category')
    
    plt.tight_layout()
    plt.savefig('temporal_model_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Main execution"""
    
    print("="*80)
    print("SIMPLIFIED TEMPORAL ROUND PREDICTION")
    print("="*80)
    
    # Load data
    data_dir = Path("clean_dataset")
    
    print("\nLoading data...")
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    round_players_df = pd.read_csv(data_dir / "round_players.csv")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    
    print(f"Loaded {len(rounds_df)} rounds from {rounds_df['match_id'].nunique()} matches")
    
    # Prepare features
    df = prepare_round_features(rounds_df, round_players_df, matches_df)
    
    # Add temporal features
    df = add_temporal_features(df)
    
    print(f"\nCreated {len(df.columns)} features")
    
    # Map-based CT advantage ratios
    rounds_with_map = rounds_df.merge(
        matches_df[['match_id', 'map_name']],
        on='match_id',
        how='left'
    )
    rounds_with_map['ct_win'] = (rounds_with_map['round_winner'] == 'ct').astype(int)
    map_ct_rate = rounds_with_map.groupby('map_name')['ct_win'].mean()
    eps = 1e-6
    map_odds_ratio = ((map_ct_rate + eps) / (1 - map_ct_rate + eps)).to_dict()
    map_odds_ratio['unknown'] = 1.0

    # Train model
    model, importance_df, scores_df = train_temporal_model(df, map_odds_ratio)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_results(importance_df, scores_df)
    
    # Save results
    print("\nSaving results...")
    importance_df.to_csv('temporal_feature_importance.csv', index=False)
    scores_df.to_csv('temporal_cv_scores.csv', index=False)
    df.to_csv('temporal_features.csv', index=False)
    print("  - temporal_model.lgb.txt")
    print("  - temporal_predictions.csv")
    print("  - temporal_feature_importance.csv")
    print("  - temporal_cv_scores.csv")
    print("  - temporal_features.csv")
    
    print("\nTemporal model complete")
    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string())
    
    print("\nKey insights:")
    print(f"- Temporal features account for {importance_df[importance_df['feature'].str.contains('lag|roll|streak')]['importance'].sum():.1%} of total importance")
    print(f"- Best lag window appears to be lag{importance_df[importance_df['feature'].str.contains('lag')].iloc[0]['feature'].split('lag')[1] if len(importance_df[importance_df['feature'].str.contains('lag')]) > 0 else 'N/A'}")
    print(f"- Model achieves {scores_df['auc'].mean():.1%} AUC in predicting round outcomes")
    if 'accuracy' in scores_df:
        print(f"- Cross-validated accuracy: {scores_df['accuracy'].mean():.1%}")


if __name__ == "__main__":
    main()
