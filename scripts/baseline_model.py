"""
Simple Baseline Model - Majority Class Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold
import json

# Configuration
DATA_DIR = Path("clean_dataset")
OUTPUT_DIR = Path("baseline_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    """Load raw round data."""
    print("="*60)
    print("SIMPLE BASELINE MODEL")
    print("="*60)
    print("\nLoading data...")
    
    rounds = pd.read_csv(DATA_DIR / "rounds.csv")
    df = rounds[['match_id', 'round_num', 'round_winner']].copy()
    
    # Convert winner to outcome
    df['outcome'] = df['round_winner'].map({'ct': 'CT_win', 't': 'T_win'})
    df = df.dropna(subset=['outcome'])
    
    print(f"Loaded {len(df):,} rounds from {df['match_id'].nunique()} matches")
    
    return df


def train_and_evaluate(train_df, test_df):
    """Train baseline and evaluate."""
    # "Training" = find most common class
    majority_class = train_df['outcome'].mode()[0]
    majority_rate = (train_df['outcome'] == majority_class).mean()
    
    # "Prediction" = always predict majority class
    y_true = (test_df['outcome'] == 'CT_win').astype(int).values
    
    if majority_class == 'CT_win':
        y_pred = np.ones(len(test_df))
        y_prob = np.ones(len(test_df)) * majority_rate
    else:
        y_pred = np.zeros(len(test_df))
        y_prob = np.ones(len(test_df)) * (1 - majority_rate)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = 0.500  # No discrimination
    logloss = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'logloss': logloss,
        'brier': brier,
        'majority_class': majority_class,
        'majority_rate': majority_rate,
    }


def cross_validate(df):
    """5-fold cross-validation."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df['match_id'].values
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        print(f"\nFold {fold}: {len(train_df):,} train, {len(test_df):,} test")
        
        fold_results = train_and_evaluate(train_df, test_df)
        fold_results['fold'] = fold
        
        print(f"  Strategy: Always predict '{fold_results['majority_class']}'")
        print(f"  Accuracy: {fold_results['accuracy']:.4f}")
        
        results.append(fold_results)
    
    return pd.DataFrame(results)


def main():
    # Load data
    df = load_data()
    
    # Show distribution
    print("\n" + "="*60)
    print("OUTCOME DISTRIBUTION")
    print("="*60)
    outcome_counts = df['outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        print(f"{outcome}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Cross-validation
    cv_results = cross_validate(df)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:    {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"AUC:         {cv_results['auc'].mean():.4f} ± {cv_results['auc'].std():.4f}")
    print(f"Log Loss:    {cv_results['logloss'].mean():.4f} ± {cv_results['logloss'].std():.4f}")
    print(f"Brier Score: {cv_results['brier'].mean():.4f} ± {cv_results['brier'].std():.4f}")
    
    # Save
    cv_results.to_csv(OUTPUT_DIR / 'baseline_cv_results.csv', index=False)
    
    metrics = {
        'model': 'Majority Class Baseline',
        'description': 'Always predicts the most common outcome',
        'strategy': f"Always predict {df['outcome'].mode()[0]}",
        'cv_mean': {
            'accuracy': float(cv_results['accuracy'].mean()),
            'auc': float(cv_results['auc'].mean()),
            'logloss': float(cv_results['logloss'].mean()),
            'brier': float(cv_results['brier'].mean()),
        },
        'cv_std': {
            'accuracy': float(cv_results['accuracy'].std()),
            'auc': float(cv_results['auc'].std()),
            'logloss': float(cv_results['logloss'].std()),
            'brier': float(cv_results['brier'].std()),
        },
        'n_folds': N_FOLDS,
        'n_rounds': len(df),
    }
    
    with open(OUTPUT_DIR / 'baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - baseline_cv_results.csv")
    print(f"  - baseline_metrics.json")
    
    print("\n" + "="*60)
    print("BASELINE COMPLETE")
    print("="*60)
    print("\nThis baseline represents the minimum performance")


if __name__ == "__main__":
    main()