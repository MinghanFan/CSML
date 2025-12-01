"""
Bayesian Network Round Winner Prediction Model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

DATA_DIR = Path("bn_analysis")
OUTPUT_DIR = Path("bn_analysis")
CPD_DIR = OUTPUT_DIR / "bn_cpd_tables"
CPD_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 13

STATE_NAMES = {
    'equip_advantage': ['T_strong', 'T_moderate', 'even', 'CT_moderate', 'CT_strong'],
    'momentum': ['T_streak', 'T_slight', 'neutral', 'CT_slight', 'CT_streak'],
    'recent_performance': ['T_performing', 'even', 'CT_performing'],
    'map_side_bias': ['T_favored', 'balanced', 'CT_favored'],
    'round_phase': ['first_half', 'second_half', 'overtime'],
    'buy_phase': ['major_advantage', 'both_full', 'both_eco', 'semi_situation'],
    'score_pressure': ['close', 'moderate', 'blowout'],
    'outcome': ['CT_win', 'T_win'],
}

FEATURES = [
    'equip_advantage',
    'momentum',
    'recent_performance',
    'map_side_bias',
    'round_phase',
    'buy_phase',
    'score_pressure',
]
TARGET = 'outcome'


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "rounds_discretized.csv"
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = FEATURES + [TARGET, 'match_id']
    if 'round_num' in df.columns:
        cols.append('round_num')
    df_bn = df[cols].copy()
    if 'round_num' in df_bn.columns:
        df_bn = df_bn[df_bn['round_num'] >= 3].drop(columns=['round_num'])
    df_bn = df_bn.dropna()
    return df_bn, FEATURES


def build_structure() -> DiscreteBayesianNetwork:
    edges = [
        # Primary predictors → outcome
        ('equip_advantage', 'outcome'),
        ('momentum', 'outcome'),
        ('recent_performance', 'outcome'),
        ('map_side_bias', 'outcome'),
        
        # Supporting structure
        ('recent_performance', 'momentum'),
        ('score_pressure', 'momentum'),
        ('round_phase', 'buy_phase'),
        ('buy_phase', 'equip_advantage'),
    ]
    model = DiscreteBayesianNetwork(edges)
    return model


def train_model(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> DiscreteBayesianNetwork:
    node_cols = list(model.nodes())
    model.fit(
        df[node_cols],
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10,
        state_names=STATE_NAMES,
    )
    return model


def predict(model: DiscreteBayesianNetwork, df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    infer = VariableElimination(model)
    preds = []
    probs = []
    for _, row in df.iterrows():
        evidence = {f: row[f] for f in features}
        result = infer.query(variables=[TARGET], evidence=evidence, show_progress=False)
        ct_prob = result.values[0] if result.state_names[TARGET][0] == 'CT_win' else result.values[1]
        preds.append('CT_win' if ct_prob > 0.5 else 'T_win')
        probs.append(ct_prob)
    return np.array(preds), np.array(probs)


def evaluate_fold(model: DiscreteBayesianNetwork, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], fold: int) -> Dict:
    trained = train_model(model, train_df)
    y_true = (test_df[TARGET] == 'CT_win').astype(int).values
    preds, probs = predict(trained, test_df, features)
    y_pred = (preds == 'CT_win').astype(int)
    return {
        'fold': fold,
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, probs),
        'logloss': log_loss(y_true, probs),
        'brier': brier_score_loss(y_true, probs),
        'n_train': len(train_df),
        'n_test': len(test_df),
    }


def cross_validate(df: pd.DataFrame, features: List[str], n_folds: int = 5) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Perform cross-validation and collect out-of-fold predictions for calibration.
    
    Returns:
        cv_results: DataFrame with per-fold metrics
        oof_predictions: Out-of-fold predicted probabilities
        oof_true: Out-of-fold true labels
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = df['match_id'].values
    base = build_structure()
    results = []
    
    # Arrays to store out-of-fold predictions
    oof_predictions = np.zeros(len(df))
    oof_true = np.zeros(len(df))
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Train and get predictions
        trained = train_model(base.copy(), train_df)
        y_true = (test_df[TARGET] == 'CT_win').astype(int).values
        preds, probs = predict(trained, test_df, features)
        y_pred = (preds == 'CT_win').astype(int)
        
        # Store out-of-fold predictions
        oof_predictions[test_idx] = probs
        oof_true[test_idx] = y_true
        
        # Calculate metrics
        res = {
            'fold': fold,
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, probs),
            'logloss': log_loss(y_true, probs),
            'brier': brier_score_loss(y_true, probs),
            'n_train': len(train_df),
            'n_test': len(test_df),
        }
        results.append(res)
    
    return pd.DataFrame(results), oof_predictions, oof_true


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    Lower is better (0 = perfect calibration).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def analyze_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Analyze calibration of predictions.
    
    Returns calibration metrics and curve data.
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )
    
    # Calculate ECE
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    
    # Calculate Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'ece': float(ece),
        'brier': float(brier),
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
    }


def save_cpds(model: DiscreteBayesianNetwork, output_dir: Path):
    for cpd in model.get_cpds():
        node = cpd.variable
        states = cpd.state_names[node]
        parents = cpd.variables[1:]
        if parents:
            from itertools import product
            parent_states = [cpd.state_names[p] for p in parents]
            combos = list(product(*parent_states))
            values = np.array(cpd.values).reshape(len(states), -1).T
            df = pd.DataFrame(values, index=[str(c) for c in combos], columns=states)
            df.index.name = 'evidence'
        else:
            values = np.array(cpd.values).reshape(1, len(states))
            df = pd.DataFrame(values, columns=states, index=['(prior)'])
        df.to_csv(output_dir / f"cpd_{node}.csv")


def inference_examples(model: DiscreteBayesianNetwork, df: pd.DataFrame) -> Dict:
    infer = VariableElimination(model)
    feature_cols = [n for n in model.nodes() if n != TARGET]
    from collections import Counter
    counts = Counter(tuple(r) for r in df[feature_cols].dropna().itertuples(index=False, name=None))
    common = counts.most_common(5)
    rare = [item for item in sorted(counts.items(), key=lambda x: x[1]) if item[1] > 0][:3]

    scenarios = {}
    for rank, (combo, count) in enumerate(common, 1):
        scenarios[f"Top-{rank} support (n={count})"] = dict(zip(feature_cols, combo))
    for rank, (combo, count) in enumerate(rare, 1):
        scenarios[f"Low-{rank} support (n={count})"] = dict(zip(feature_cols, combo))

    results = {}
    for name, ev in scenarios.items():
        res = infer.query(variables=[TARGET], evidence=ev, show_progress=False)
        ct_prob = res.values[0] if res.state_names[TARGET][0] == 'CT_win' else res.values[1]
        t_prob = 1 - ct_prob
        support = int(((df[feature_cols] == pd.Series(ev)).all(axis=1)).sum())
        results[name] = {
            'evidence': ev,
            'ct_win_prob': float(ct_prob),
            't_win_prob': float(t_prob),
            'prediction': 'CT_win' if ct_prob > 0.5 else 'T_win',
            'support': support,
        }
    return results


def main():
    print("="*80)
    print("BAYESIAN NETWORK TRAINING WITH CALIBRATION ANALYSIS")
    print("="*80)
    
    # Load and prepare data
    df_raw = load_data()
    df_clean, features = prepare_data(df_raw)
    
    print(f"\nTraining on {len(df_clean):,} rounds")
    print(f"Features: {len(features)}")
    
    # Cross-validation with out-of-fold predictions
    print(f"\nPerforming {N_FOLDS}-fold cross-validation...")
    cv, oof_predictions, oof_true = cross_validate(df_clean, features, n_folds=N_FOLDS)
    cv.to_csv(OUTPUT_DIR / 'bn_cv_results.csv', index=False)
    
    # Print CV results
    print("\nCross-validation results:")
    print(f"  Accuracy:  {cv['accuracy'].mean():.4f} ± {cv['accuracy'].std():.4f}")
    print(f"  AUC:       {cv['auc'].mean():.4f} ± {cv['auc'].std():.4f}")
    print(f"  Log Loss:  {cv['logloss'].mean():.4f} ± {cv['logloss'].std():.4f}")
    print(f"  Brier:     {cv['brier'].mean():.4f} ± {cv['brier'].std():.4f}")

    # After cross_validate() returns oof_predictions, oof_true
    print("\nChecking for prediction patterns...")
    print(f"Min prediction: {oof_predictions.min()}")
    print(f"Max prediction: {oof_predictions.max()}")
    print(f"Unique predictions: {len(np.unique(oof_predictions))}")

    # Check if predictions are too concentrated
    hist, bins = np.histogram(oof_predictions, bins=10)
    print(f"Distribution across bins: {hist}")

    # Check for extreme confidence
    extreme_confident = ((oof_predictions < 0.1) | (oof_predictions > 0.9)).sum()
    print(f"Extreme predictions (<0.1 or >0.9): {extreme_confident}/{len(oof_predictions)} ({extreme_confident/len(oof_predictions)*100:.1f}%)")
    
    # Calibration analysis
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS")
    print("="*80)
    
    calibration_results = analyze_calibration(oof_true, oof_predictions)
    
    print(f"\nCalibration Metrics:")
    print(f"  Expected Calibration Error (ECE): {calibration_results['ece']:.4f}")
    print(f"  Brier Score:                       {calibration_results['brier']:.4f}")
    
    ece = calibration_results['ece']
    
    # Train final model on full data
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    
    model = build_structure()
    model = train_model(model, df_clean)
    
    # Save CPDs
    save_cpds(model, CPD_DIR)
    print(f"\nCPDs saved to {CPD_DIR}")
    
    # Save model
    joblib.dump({
        'model': model, 
        'features': features, 
        'structure': list(model.edges()),
        'oof_predictions': oof_predictions,
        'oof_true': oof_true,
    }, OUTPUT_DIR / 'bn_model.pkl')
    print(f"Model saved to {OUTPUT_DIR / 'bn_model.pkl'}")
    
    # Generate inference examples
    examples = inference_examples(model, df_clean)
    with open(OUTPUT_DIR / 'bn_inference_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Inference examples saved to {OUTPUT_DIR / 'bn_inference_examples.json'}")
    
    # Save metrics with calibration
    metrics = {
        'cv_mean': {
            'accuracy': float(cv['accuracy'].mean()),
            'auc': float(cv['auc'].mean()),
            'logloss': float(cv['logloss'].mean()),
            'brier': float(cv['brier'].mean()),
        },
        'cv_std': {
            'accuracy': float(cv['accuracy'].std()),
            'auc': float(cv['auc'].std()),
            'logloss': float(cv['logloss'].std()),
            'brier': float(cv['brier'].std()),
        },
        'calibration': {
            'ece': calibration_results['ece'],
            'brier': calibration_results['brier'],
            'fraction_of_positives': calibration_results['fraction_of_positives'],
            'mean_predicted_value': calibration_results['mean_predicted_value'],
        },
        'n_folds': N_FOLDS,
        'n_rounds': len(df_clean),
        'structure': {'nodes': len(model.nodes()), 'edges': len(model.edges())},
    }
    
    with open(OUTPUT_DIR / 'bn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {OUTPUT_DIR / 'bn_metrics.json'}")
    
    print("\n" + "="*80)
    print("BAYESIAN NETWORK TRAINING COMPLETE")
    print("="*80)
    print("\nKey Results:")
    print(f"  - Accuracy:  {metrics['cv_mean']['accuracy']:.1%}")
    print(f"  - AUC:       {metrics['cv_mean']['auc']:.4f}")
    print(f"  - ECE:       {calibration_results['ece']:.4f}")
    print(f"\nCalibration Status: ", end="")
    if ece < 0.05:
        print("Excellent")
    elif ece < 0.10:
        print("Good")
    else:
        print("Needs improvement")


if __name__ == "__main__":
    main()