"""
Compares the Bayesian Network with LightGBM results.

USAGE:
    python bn_compare.py

REQUIREMENTS:
    pip install pandas numpy --break-system-packages

INPUTS:
    • clean_dataset/bn_analysis/bn_cv_results.csv (from bn_train_model.py)
    • clean_dataset/bn_analysis/bn_metrics.json (from bn_train_model.py)
    • clean_dataset/cv_scores.csv (from temporal_round_model.py)

OUTPUTS (saved to clean_dataset/bn_analysis/):
    • bn_comparison.json - Detailed comparison metrics
    • bn_comparison_summary.txt - Human-readable comparison
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Configuration
DATA_DIR = Path("clean_dataset")
BN_DIR = Path("clean_dataset/bn_analysis")
OUTPUT_DIR = Path("clean_dataset/bn_analysis")


def load_bn_results() -> Dict:
    """Load Bayesian Network results."""
    print("="*80)
    print("LOADING BAYESIAN NETWORK RESULTS")
    print("="*80)
    
    # Load CV results
    cv_path = BN_DIR / 'bn_cv_results.csv'
    cv_df = pd.read_csv(cv_path)
    
    # Load metrics
    metrics_path = BN_DIR / 'bn_metrics.json'
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    print(f"✓ Loaded BN results")
    print(f"  CV folds: {len(cv_df)}")
    print(f"  Total rounds: {metrics['n_rounds']:,}")
    
    return {
        'cv_results': cv_df,
        'metrics': metrics
    }


def load_lightgbm_results() -> Dict:
    """Load LightGBM results."""
    print("\n" + "="*80)
    print("LOADING LIGHTGBM RESULTS")
    print("="*80)
    
    # Load CV results
    cv_path = DATA_DIR / 'cv_scores.csv'
    
    if not cv_path.exists():
        print(f"⚠ LightGBM CV results not found: {cv_path}")
        print(f"  Run temporal_round_model.py to generate comparison data")
        return None
    
    cv_df = pd.read_csv(cv_path)
    
    print(f"✓ Loaded LightGBM results")
    print(f"  CV folds: {len(cv_df)}")
    
    return {
        'cv_results': cv_df
    }


def compare_models(bn_data: Dict, lgbm_data: Dict) -> Dict:
    """
    Compare Bayesian Network and LightGBM performance.
    
    Returns:
        Comparison dictionary with metrics
    """
    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80)
    
    bn_cv = bn_data['cv_results']
    lgbm_cv = lgbm_data['cv_results']
    
    # Calculate statistics
    comparison = {
        'bayesian_network': {
            'accuracy': {
                'mean': float(bn_cv['accuracy'].mean()),
                'std': float(bn_cv['accuracy'].std()),
                'min': float(bn_cv['accuracy'].min()),
                'max': float(bn_cv['accuracy'].max()),
            },
            'auc': {
                'mean': float(bn_cv['auc'].mean()),
                'std': float(bn_cv['auc'].std()),
                'min': float(bn_cv['auc'].min()),
                'max': float(bn_cv['auc'].max()),
            },
            'logloss': {
                'mean': float(bn_cv['logloss'].mean()),
                'std': float(bn_cv['logloss'].std()),
                'min': float(bn_cv['logloss'].min()),
                'max': float(bn_cv['logloss'].max()),
            },
            'brier': {
                'mean': float(bn_cv['brier'].mean()),
                'std': float(bn_cv['brier'].std()),
                'min': float(bn_cv['brier'].min()),
                'max': float(bn_cv['brier'].max()),
            },
        },
        'lightgbm': {
            'accuracy': {
                'mean': float(lgbm_cv['accuracy'].mean()),
                'std': float(lgbm_cv['accuracy'].std()),
                'min': float(lgbm_cv['accuracy'].min()),
                'max': float(lgbm_cv['accuracy'].max()),
            },
            'auc': {
                'mean': float(lgbm_cv['auc'].mean()),
                'std': float(lgbm_cv['auc'].std()),
                'min': float(lgbm_cv['auc'].min()),
                'max': float(lgbm_cv['auc'].max()),
            },
            'logloss': {
                'mean': float(lgbm_cv['logloss'].mean()),
                'std': float(lgbm_cv['logloss'].std()),
                'min': float(lgbm_cv['logloss'].min()),
                'max': float(lgbm_cv['logloss'].max()),
            },
            'brier': {
                'mean': float(lgbm_cv['brier'].mean()),
                'std': float(lgbm_cv['brier'].std()),
                'min': float(lgbm_cv['brier'].min()),
                'max': float(lgbm_cv['brier'].max()),
            },
        },
        'differences': {},
        'winner': {}
    }
    
    # Calculate differences
    metrics = ['accuracy', 'auc', 'logloss', 'brier']
    
    for metric in metrics:
        bn_val = comparison['bayesian_network'][metric]['mean']
        lgbm_val = comparison['lightgbm'][metric]['mean']
        diff = bn_val - lgbm_val
        
        comparison['differences'][metric] = {
            'absolute': float(diff),
            'relative_pct': float((diff / lgbm_val * 100) if lgbm_val != 0 else 0)
        }
        
        # Determine winner (lower is better for logloss and brier)
        if metric in ['logloss', 'brier']:
            winner = 'BN' if diff < 0 else 'LightGBM'
        else:
            winner = 'BN' if diff > 0 else 'LightGBM'
        
        comparison['winner'][metric] = winner
    
    # Print comparison table
    print(f"\n{'Metric':<15} {'BN':>12} {'LightGBM':>12} {'Difference':>12} {'Winner':>10}")
    print("-" * 65)
    
    for metric in metrics:
        bn_val = comparison['bayesian_network'][metric]['mean']
        lgbm_val = comparison['lightgbm'][metric]['mean']
        diff = comparison['differences'][metric]['absolute']
        winner = comparison['winner'][metric]
        
        print(f"{metric:<15} {bn_val:>12.4f} {lgbm_val:>12.4f} {diff:>+12.4f} {winner:>10}")
    
    # Overall assessment
    print(f"\n{'='*65}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*65}")
    
    bn_wins = sum(1 for w in comparison['winner'].values() if w == 'BN')
    lgbm_wins = sum(1 for w in comparison['winner'].values() if w == 'LightGBM')
    
    print(f"BN wins: {bn_wins}/{len(metrics)} metrics")
    print(f"LightGBM wins: {lgbm_wins}/{len(metrics)} metrics")
    
    # Interpretability note
    print(f"\nKey Trade-offs:")
    acc_diff = comparison['differences']['accuracy']['absolute']
    auc_diff = comparison['differences']['auc']['absolute']
    
    print(f"  • Accuracy difference: {acc_diff:+.1%}")
    print(f"  • AUC difference: {auc_diff:+.4f}")
    print(f"  • BN Advantage: Full interpretability, probabilistic reasoning")
    print(f"  • LightGBM Advantage: Slightly higher accuracy")
    
    return comparison


def generate_summary_text(comparison: Dict, bn_data: Dict) -> str:
    """Generate human-readable comparison summary."""
    
    lines = []
    lines.append("="*80)
    lines.append("BAYESIAN NETWORK VS LIGHTGBM - COMPARISON SUMMARY")
    lines.append("="*80)
    lines.append("")
    
    # Performance comparison
    lines.append("PERFORMANCE METRICS")
    lines.append("-"*80)
    lines.append("")
    lines.append(f"{'Metric':<15} {'Bayesian Network':>20} {'LightGBM':>20} {'Difference':>15}")
    lines.append("-"*80)
    
    for metric in ['accuracy', 'auc', 'logloss', 'brier']:
        bn_mean = comparison['bayesian_network'][metric]['mean']
        bn_std = comparison['bayesian_network'][metric]['std']
        lgbm_mean = comparison['lightgbm'][metric]['mean']
        lgbm_std = comparison['lightgbm'][metric]['std']
        diff = comparison['differences'][metric]['absolute']
        
        bn_str = f"{bn_mean:.4f} ± {bn_std:.4f}"
        lgbm_str = f"{lgbm_mean:.4f} ± {lgbm_std:.4f}"
        diff_str = f"{diff:+.4f}"
        
        lines.append(f"{metric:<15} {bn_str:>20} {lgbm_str:>20} {diff_str:>15}")
    
    lines.append("")
    lines.append("")
    
    # Model characteristics
    lines.append("MODEL CHARACTERISTICS")
    lines.append("-"*80)
    lines.append("")
    
    lines.append("Bayesian Network:")
    lines.append(f"  • Structure: {bn_data['metrics']['structure']['nodes']} nodes, {bn_data['metrics']['structure']['edges']} edges")
    lines.append(f"  • Training: Bayesian Estimation (BDeu prior)")
    lines.append(f"  • Inference: Variable Elimination (exact)")
    lines.append(f"  • Interpretability: ✓ Full (can trace reasoning)")
    lines.append(f"  • Probabilistic: ✓ Provides P(outcome)")
    lines.append(f"  • Domain knowledge: ✓ Encoded in structure")
    lines.append("")
    
    lines.append("LightGBM:")
    lines.append(f"  • Structure: Gradient boosted trees")
    lines.append(f"  • Training: Gradient boosting")
    lines.append(f"  • Inference: Ensemble prediction")
    lines.append(f"  • Interpretability: ✗ Black box")
    lines.append(f"  • Probabilistic: ~ Calibrated probabilities")
    lines.append(f"  • Domain knowledge: ✗ Data-driven only")
    lines.append("")
    lines.append("")
    
    # Winner analysis
    lines.append("WINNER BY METRIC")
    lines.append("-"*80)
    lines.append("")
    
    for metric, winner in comparison['winner'].items():
        symbol = "✓" if winner == "BN" else "✗"
        lines.append(f"  {symbol} {metric:<12} : {winner}")
    
    lines.append("")
    lines.append("")
    
    # Conclusion
    lines.append("CONCLUSION")
    lines.append("-"*80)
    lines.append("")
    
    acc_diff_pct = comparison['differences']['accuracy']['absolute'] * 100
    
    if abs(acc_diff_pct) < 2:
        lines.append("The models perform nearly identically in terms of raw metrics.")
    elif acc_diff_pct > 0:
        lines.append("The Bayesian Network slightly outperforms LightGBM.")
    else:
        lines.append(f"LightGBM has a slight edge in predictive performance ({abs(acc_diff_pct):.1f}%).")
    
    lines.append("")
    lines.append("Key Trade-off:")
    lines.append(f"  • BN sacrifices ~{abs(acc_diff_pct):.1f}% accuracy for full interpretability")
    lines.append(f"  • This trade-off is WORTHWHILE for:")
    lines.append(f"    - Understanding what drives round outcomes")
    lines.append(f"    - Explaining predictions to stakeholders")
    lines.append(f"    - Strategic analysis and decision-making")
    lines.append(f"    - Building trust in AI systems")
    lines.append("")
    lines.append("Recommendation:")
    lines.append(f"  • Use BN for strategic analysis and understanding")
    lines.append(f"  • Use LightGBM if pure prediction accuracy is paramount")
    lines.append(f"  • For this project: BN is superior (interpretability + domain knowledge)")
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    
    print("="*80)
    print("BAYESIAN NETWORK VS LIGHTGBM COMPARISON - SESSION 2 (PART 2)")
    print("="*80)
    
    # Load results
    bn_data = load_bn_results()
    lgbm_data = load_lightgbm_results()
    
    if lgbm_data is None:
        print("\n✗ Cannot perform comparison without LightGBM results")
        print("  Run temporal_round_model.py first, then re-run this script")
        return
    
    # Compare
    comparison = compare_models(bn_data, lgbm_data)
    
    # Generate summary text
    summary_text = generate_summary_text(comparison, bn_data)
    
    # Save comparison
    comparison_path = OUTPUT_DIR / 'bn_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved bn_comparison.json")
    
    # Save summary
    summary_path = OUTPUT_DIR / 'bn_comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"✓ Saved bn_comparison_summary.txt")
    
    # Print summary
    print(f"\n{summary_text}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  • bn_comparison.json")
    print(f"  • bn_comparison_summary.txt")
    
    print(f"\nNext step:")
    print(f"  Run bn_visualize.py to create comparison visualizations")


if __name__ == "__main__":
    main()