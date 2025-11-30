"""
Bayesian Network Model Training 

USAGE:
    python bn_train_model.py

REQUIREMENTS:
    pip install pgmpy scikit-learn joblib --break-system-packages

INPUTS (from Session 1):
    • clean_dataset/bn_analysis/rounds_discretized.csv

OUTPUTS (saved to clean_dataset/bn_analysis/):
    • bn_model.pkl - Trained Bayesian Network
    • bn_cv_results.csv - Cross-validation scores per fold
    • bn_metrics.json - Overall performance metrics
    • bn_cpd_tables/ - Conditional Probability Distribution tables
    • bn_inference_examples.json - Scenario predictions

STRUCTURE (8 nodes):
    Nodes: equip_advantage, momentum, recent_performance, map_side_bias,
           round_phase, buy_phase, score_pressure, outcome
    
    Edges:
      - equip_advantage → outcome
      - momentum → outcome  
      - recent_performance → outcome
      - map_side_bias → outcome
      - recent_performance → momentum
      - round_phase → buy_phase
      - buy_phase → equip_advantage
      - score_pressure → momentum
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
DATA_DIR = Path("clean_dataset/bn_analysis")
OUTPUT_DIR = Path("clean_dataset/bn_analysis")
CPD_DIR = OUTPUT_DIR / "bn_cpd_tables"
CPD_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
N_FOLDS = 5
RANDOM_STATE = 42
USE_BAYESIAN_ESTIMATION = True  # Use BDeu prior (more robust than MLE)

# Explicit state spaces to keep inference aligned with discretization bins
STATE_NAMES = {
    'equip_advantage': ['T_strong', 'T_moderate', 'even', 'CT_moderate', 'CT_strong'],
    'momentum': ['T_streak', 'T_slight', 'neutral', 'CT_slight', 'CT_streak'],
    'recent_performance': ['T_performing', 'even', 'CT_performing'],
    'map_side_bias': ['T_favored', 'balanced', 'CT_favored'],
    'round_phase': ['first_half', 'second_half', 'overtime'],
    'buy_phase': ['unknown', 'major_advantage', 'both_full', 'both_eco', 'semi_situation', 'mixed'],
    'score_pressure': ['close', 'moderate', 'blowout'],
    'outcome': ['CT_win', 'T_win'],
}


def load_data() -> pd.DataFrame:
    """Load discretized data from Session 1."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_path = DATA_DIR / "rounds_discretized.csv"
    df = pd.read_csv(data_path)
    
    print(f"✓ Loaded {len(df):,} rounds")
    print(f"✓ Columns: {len(df.columns)}")
    
    return df


def prepare_bn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for Bayesian Network training.
    
    Returns:
        Clean dataframe and list of feature columns
    """
    print("\n" + "="*80)
    print("PREPARING DATA FOR BAYESIAN NETWORK")
    print("="*80)
    
    # BN feature columns (8 nodes total, 7 features + 1 target)
    bn_features = [
        'equip_advantage',
        'momentum',
        'recent_performance',
        'map_side_bias',
        'round_phase',
        'buy_phase',
        'score_pressure',
    ]
    
    target = 'outcome'
    
    # Select columns and keep round_num if available for filtering
    required_cols = bn_features + [target, 'match_id']
    if 'round_num' in df.columns:
        required_cols.append('round_num')

    df_bn = df[required_cols].copy()

    # Filter to rounds with minimum history (round 3+)
    # This ensures momentum and recent_performance have meaningful values
    if 'round_num' in df_bn.columns:
        df_clean = df_bn[df_bn['round_num'] >= 3].copy()
        df_clean = df_clean.drop(columns=['round_num'])
    else:
        df_clean = df_bn.copy()

    # Remove rows with missing values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    
    print(f"✓ Selected {len(bn_features)} features + target")
    print(f"✓ Removed missing values: {initial_count - len(df_clean):,} rows")
    print(f"✓ Clean dataset: {len(df_clean):,} rounds")
    print(f"\nFeatures: {', '.join(bn_features)}")
    print(f"Target: {target}")
    
    # Check class balance
    outcome_dist = df_clean[target].value_counts()
    print(f"\nOutcome distribution:")
    for outcome, count in outcome_dist.items():
        print(f"  {outcome}: {count:,} ({count/len(df_clean)*100:.1f}%)")
    
    return df_clean, bn_features


def build_bn_structure() -> DiscreteBayesianNetwork:
    """
    Build Bayesian Network structure (8-node complex model).
    
    Structure based on CS:GO domain knowledge and Session 1 independence tests:
    - Top predictors (momentum, equipment, recent_performance) → outcome
    - Causal relationships (recent_performance → momentum)
    - Economic cycles (round_phase → buy_phase → equip_advantage)
    - Psychological effects (score_pressure → momentum)
    """
    print("\n" + "="*80)
    print("BUILDING BAYESIAN NETWORK STRUCTURE")
    print("="*80)
    
    # Define edges (parent → child)
    edges = [
        # Primary predictors → outcome
        ('equip_advantage', 'outcome'),
        ('momentum', 'outcome'),
        ('recent_performance', 'outcome'),
        ('map_side_bias', 'outcome'),
        
        # Causal relationships between features
        ('recent_performance', 'momentum'),  # Performance creates confidence
        ('round_phase', 'buy_phase'),        # Economy cycles by half
        ('buy_phase', 'equip_advantage'),    # Buy decisions → equipment
        ('score_pressure', 'momentum'),      # Pressure affects psychology
    ]
    
    model = DiscreteBayesianNetwork(edges)
    
    print(f"✓ Created Bayesian Network")
    print(f"  Nodes: {len(model.nodes())}")
    print(f"  Edges: {len(model.edges())}")
    print(f"\nEdge structure:")
    for parent, child in edges:
        print(f"  {parent:25s} → {child}")
    
    # Validate acyclicity only (full CPD validation happens after fitting)
    try:
        if hasattr(model, 'is_dag') and not model.is_dag():
            raise ValueError("Graph contains cycles")
        print(f"\n✓ Structure validated (acyclic)")
    except Exception as e:
        print(f"\n⚠ Warning: Could not validate structure: {e}")
    
    return model


def train_bn(
    model: DiscreteBayesianNetwork,
    df: pd.DataFrame,
    use_bayesian: bool = True
) -> DiscreteBayesianNetwork:
    """
    Train Bayesian Network using Bayesian estimation.
    
    Args:
        model: BayesianNetwork structure
        df: Training data
        use_bayesian: If True, use Bayesian estimation with BDeu prior
    
    Returns:
        Trained BayesianNetwork with CPDs
    """
    # Use only nodes defined in the model (drop helper columns like match_id)
    node_cols = list(model.nodes())
    df_nodes = df[node_cols].copy()

    if use_bayesian:
        model.fit(
            df_nodes,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_sample_size=10,
            state_names=STATE_NAMES,
        )
    else:
        model.fit(df_nodes, state_names=STATE_NAMES)
    
    return model


def save_cpd_tables(model: DiscreteBayesianNetwork, output_dir: Path):
    """Save Conditional Probability Distribution tables as CSV."""
    print("\n" + "="*80)
    print("SAVING CPD TABLES")
    print("="*80)
    
    for cpd in model.get_cpds():
        node_name = cpd.variable
        node_states = cpd.state_names[cpd.variable]
        parent_vars = cpd.variables[1:]

        if parent_vars:
            from itertools import product

            parent_state_lists = [cpd.state_names[var] for var in parent_vars]
            evidence_combos = list(product(*parent_state_lists))

            # pgmpy flattens CPD with child states first; reshape to (rows=evidence, cols=child states)
            values = np.array(cpd.values).reshape(len(node_states), -1).T
            cpd_df = pd.DataFrame(
                values,
                index=[str(combo) for combo in evidence_combos],
                columns=node_states,
            )
            cpd_df.index.name = 'evidence'
        else:
            values = np.array(cpd.values).reshape(1, len(node_states))
            cpd_df = pd.DataFrame(values, columns=node_states, index=['(prior)'])

        cpd_path = output_dir / f"cpd_{node_name}.csv"
        cpd_df.to_csv(cpd_path)
        print(f"  ✓ Saved CPD for '{node_name}' (shape: {cpd_df.shape})")
    
    print(f"\n✓ All CPDs saved to {output_dir}")


def predict_bn(
    model: DiscreteBayesianNetwork,
    df: pd.DataFrame,
    features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using Variable Elimination inference.
    
    Returns:
        predictions: Class labels (CT_win or T_win)
        probabilities: P(CT_win) for each sample
    """
    inference = VariableElimination(model)
    
    predictions = []
    probabilities = []
    
    for idx, row in df.iterrows():
        # Prepare evidence (all features)
        evidence = {feat: row[feat] for feat in features}
        
        # Query outcome
        result = inference.query(
            variables=['outcome'],
            evidence=evidence,
            show_progress=False
        )
        
        # Get probabilities
        # pgmpy orders states alphabetically, so CT_win comes before T_win
        ct_prob = result.values[0] if result.state_names['outcome'][0] == 'CT_win' else result.values[1]
        
        # Prediction
        pred = 'CT_win' if ct_prob > 0.5 else 'T_win'
        
        predictions.append(pred)
        probabilities.append(ct_prob)
    
    return np.array(predictions), np.array(probabilities)


def evaluate_bn_fold(
    model: DiscreteBayesianNetwork,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    fold_num: int
) -> Dict:
    """Train and evaluate BN on a single fold."""
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}")
    print(f"{'='*60}")
    print(f"Train: {len(train_df):,} rounds | Test: {len(test_df):,} rounds")
    
    # Train
    print("\nTraining model...")
    trained_model = train_bn(model, train_df, use_bayesian=USE_BAYESIAN_ESTIMATION)
    
    # Predict
    print("Making predictions...")
    predictions, probabilities = predict_bn(trained_model, test_df, features)
    
    # Convert true labels to binary
    y_true = (test_df['outcome'] == 'CT_win').astype(int).values
    y_pred = (predictions == 'CT_win').astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probabilities)
    logloss = log_loss(y_true, probabilities)
    brier = brier_score_loss(y_true, probabilities)
    
    print(f"\nResults:")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  AUC:          {auc:.4f}")
    print(f"  Log Loss:     {logloss:.4f}")
    print(f"  Brier Score:  {brier:.4f}")
    
    return {
        'fold': fold_num,
        'accuracy': accuracy,
        'auc': auc,
        'logloss': logloss,
        'brier': brier,
        'n_train': len(train_df),
        'n_test': len(test_df)
    }


def cross_validate_bn(
    df: pd.DataFrame,
    features: List[str],
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Perform cross-validation on Bayesian Network.
    
    Uses GroupKFold to ensure rounds from same match stay together.
    """
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION ({n_folds} FOLDS)")
    print("="*80)
    
    # Group by match_id
    groups = df['match_id'].values
    
    # Initialize cross-validator
    gkf = GroupKFold(n_splits=n_folds)
    
    # Store results
    cv_results = []
    
    # Build base structure (reused for each fold)
    base_model = build_bn_structure()
    
    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Evaluate fold
        fold_result = evaluate_bn_fold(
            base_model.copy(),  # Fresh copy for each fold
            train_df,
            test_df,
            features,
            fold_num
        )
        
        cv_results.append(fold_result)
    
    # Convert to DataFrame
    cv_df = pd.DataFrame(cv_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Accuracy:    {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"AUC:         {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    print(f"Log Loss:    {cv_df['logloss'].mean():.4f} ± {cv_df['logloss'].std():.4f}")
    print(f"Brier Score: {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}")
    
    return cv_df


def train_final_model(
    df: pd.DataFrame,
    features: List[str]
) -> DiscreteBayesianNetwork:
    """Train final model on full dataset."""
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*80)
    
    model = build_bn_structure()
    model = train_bn(model, df, use_bayesian=USE_BAYESIAN_ESTIMATION)
    
    print(f"\n✓ Final model trained on {len(df):,} rounds")
    
    return model


def generate_inference_examples(model: DiscreteBayesianNetwork) -> Dict:
    """
    Generate scenario-based predictions using the trained BN.
    
    Returns dict of scenarios with predictions.
    """
    print("\n" + "="*80)
    print("GENERATING INFERENCE EXAMPLES")
    print("="*80)
    
    inference = VariableElimination(model)
    
    scenarios = {
        "CT Full Buy Advantage": {
            'equip_advantage': 'CT_strong',
            'momentum': 'neutral',
            'recent_performance': 'even',
            'map_side_bias': 'balanced',
            'round_phase': 'first_half',
            'buy_phase': 'both_full',
            'score_pressure': 'close'
        },
        "T Full Buy Advantage": {
            'equip_advantage': 'T_strong',
            'momentum': 'neutral',
            'recent_performance': 'even',
            'map_side_bias': 'balanced',
            'round_phase': 'first_half',
            'buy_phase': 'both_full',
            'score_pressure': 'close'
        },
        "Even Match with CT Momentum": {
            'equip_advantage': 'even',
            'momentum': 'CT_streak',
            'recent_performance': 'CT_performing',
            'map_side_bias': 'balanced',
            'round_phase': 'second_half',
            'buy_phase': 'both_full',
            'score_pressure': 'moderate'
        },
        "T Comeback Pressure": {
            'equip_advantage': 'even',
            'momentum': 'T_streak',
            'recent_performance': 'T_performing',
            'map_side_bias': 'balanced',
            'round_phase': 'second_half',
            'buy_phase': 'both_full',
            'score_pressure': 'blowout'
        },
        "Overtime - CT Favored Map": {
            'equip_advantage': 'CT_moderate',
            'momentum': 'neutral',
            'recent_performance': 'even',
            'map_side_bias': 'CT_favored',
            'round_phase': 'overtime',
            'buy_phase': 'both_full',
            'score_pressure': 'moderate'
        }
    }
    
    results = {}
    
    for scenario_name, evidence in scenarios.items():
        result = inference.query(
            variables=['outcome'],
            evidence=evidence,
            show_progress=False
        )
        
        # Get probabilities
        ct_prob = result.values[0] if result.state_names['outcome'][0] == 'CT_win' else result.values[1]
        t_prob = 1 - ct_prob
        
        prediction = 'CT_win' if ct_prob > 0.5 else 'T_win'
        confidence = max(ct_prob, t_prob)
        
        results[scenario_name] = {
            'evidence': evidence,
            'ct_win_prob': float(ct_prob),
            't_win_prob': float(t_prob),
            'prediction': prediction,
            'confidence': float(confidence)
        }
        
        print(f"\n📊 {scenario_name}")
        print(f"   → CT Win: {ct_prob:.1%}")
        print(f"   → T Win:  {t_prob:.1%}")
        print(f"   → Prediction: {prediction} (confidence: {confidence:.1%})")
    
    return results


def main():
    """Main execution."""
    
    print("="*80)
    print("BAYESIAN NETWORK MODEL TRAINING - SESSION 2 (PART 1)")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Prepare for BN
    df_clean, features = prepare_bn_data(df)
    
    # Cross-validation
    cv_results = cross_validate_bn(df_clean, features, n_folds=N_FOLDS)
    
    # Save CV results
    cv_results.to_csv(OUTPUT_DIR / 'bn_cv_results.csv', index=False)
    print(f"\n✓ Saved bn_cv_results.csv")
    
    # Train final model
    final_model = train_final_model(df_clean, features)
    
    # Save CPD tables
    save_cpd_tables(final_model, CPD_DIR)
    
    # Save model
    model_path = OUTPUT_DIR / 'bn_model.pkl'
    joblib.dump({
        'model': final_model,
        'features': features,
        'structure': list(final_model.edges())
    }, model_path)
    print(f"\n✓ Saved bn_model.pkl")
    
    # Generate inference examples
    inference_examples = generate_inference_examples(final_model)
    
    # Save inference examples
    with open(OUTPUT_DIR / 'bn_inference_examples.json', 'w') as f:
        json.dump(inference_examples, f, indent=2)
    print(f"\n✓ Saved bn_inference_examples.json")
    
    # Save overall metrics
    metrics = {
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
        'n_rounds': len(df_clean),
        'structure': {
            'nodes': len(final_model.nodes()),
            'edges': len(final_model.edges()),
        }
    }
    
    with open(OUTPUT_DIR / 'bn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved bn_metrics.json")
    
    # Final summary
    print("\n" + "="*80)
    print("PART 1 COMPLETE - MODEL TRAINING")
    print("="*80)
    print(f"\nBayesian Network Performance:")
    print(f"  Accuracy:     {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"  AUC:          {cv_results['auc'].mean():.4f} ± {cv_results['auc'].std():.4f}")
    print(f"  Log Loss:     {cv_results['logloss'].mean():.4f} ± {cv_results['logloss'].std():.4f}")
    print(f"  Brier Score:  {cv_results['brier'].mean():.4f} ± {cv_results['brier'].std():.4f}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  • bn_model.pkl")
    print(f"  • bn_cv_results.csv")
    print(f"  • bn_metrics.json")
    print(f"  • bn_inference_examples.json")
    print(f"  • bn_cpd_tables/ (8 files)")
    
    print(f"\nNext steps:")
    print(f"  1. Run bn_compare.py to compare with LightGBM")
    print(f"  2. Run bn_visualize.py to create visualizations")


if __name__ == "__main__":
    main()
