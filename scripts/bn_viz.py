"""
Creates visualizations for the Bayesian Network
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)

# Configuration
BN_DIR = Path("bn_analysis")
OUTPUT_DIR = Path("bn_analysis")

# Palette (kept as-is per your preference)
palette = sns.color_palette("tab10", 10)
COLORS = {
    "primary": palette[0],
    "secondary": palette[1],
    "tertiary": palette[2],
    "accent": palette[9],
    "grey": palette[7],
}

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=palette[:3])
sns.set_palette("tab10")
CONF_CMAP = LinearSegmentedColormap.from_list("primary_grad", ["#ffffff", COLORS["primary"]])


def load_model():
    """Load trained Bayesian Network model."""
    print("="*80)
    print("LOADING BAYESIAN NETWORK MODEL")
    print("="*80)
    
    model_path = BN_DIR / 'bn_model.pkl'
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    features = model_data['features']
    structure = model_data['structure']
    oof_true = np.asarray(model_data.get('oof_true'))
    oof_pred = np.asarray(model_data.get('oof_predictions'))
    
    print(f"Loaded BN model")
    print(f"  Nodes: {len(model.nodes())}")
    print(f"  Edges: {len(model.edges())}")
    print(f"  Features: {len(features)}")
    
    return model, features, structure, oof_true, oof_pred


def visualize_network_structure(model):
    """Visualize the Bayesian Network structure."""
    print("\n" + "="*80)
    print("CREATING NETWORK STRUCTURE VISUALIZATION")
    print("="*80)
    
    import networkx as nx

    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Color nodes by role
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == 'outcome':
            node_colors.append(COLORS["primary"])  
            node_sizes.append(9000)
        elif node in ['equip_advantage', 'momentum', 'recent_performance']:
            node_colors.append(COLORS["secondary"]) 
            node_sizes.append(9000)
        else:
            node_colors.append(COLORS["tertiary"]) 
            node_sizes.append(9000)
    
    # Draw nodes as squares instead of circles
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        alpha=0.5,
        node_size=node_sizes,
        ax=ax,
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_family='sans-serif',
        font_color='black',
        ax=ax
    )
    
    # Draw directed edges; keep arrowheads outside the larger nodes
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=COLORS["grey"],
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=2,
        connectionstyle="arc3,rad=0.1",
        # Extra margins keep arrowheads outside the large colored disks
        min_source_margin=60,
        min_target_margin=60,
        ax=ax,
    )
    
    # Title
    ax.set_title(
        'Bayesian Network Structure for CS2 Round Prediction\n',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["primary"], edgecolor='black', linewidth=2, label='Outcome'),
        Patch(facecolor=COLORS["secondary"], edgecolor='black', linewidth=2, label='Primary Predictors'),
        Patch(facecolor=COLORS["tertiary"], edgecolor='black', linewidth=2, label='Supporting Features')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bn_structure.png', dpi=300, bbox_inches='tight')
    print("Saved bn_structure.png")
    plt.close()

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color=COLORS["primary"], label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=0.8, color=COLORS["grey"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bn_roc_curve.png", dpi=200)
    plt.close()


def plot_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, color=COLORS["primary"], label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bn_pr_curve.png", dpi=200)
    plt.close()


def plot_calibration(y_true, y_prob, bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", color=COLORS["primary"], label="Model")
    plt.plot([0, 1], [0, 1], "--", lw=0.8, color=COLORS["grey"], label="Perfect")
    plt.xlabel("Predicted probability (CT win)")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bn_calibration_curve.png", dpi=200)
    plt.close()


def plot_confusion(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap=CONF_CMAP)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred T", "Pred CT"])
    plt.yticks([0, 1], ["True T", "True CT"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bn_confusion_matrix.png", dpi=200)
    plt.close()


def plot_extremes(y_prob, low=0.1, high=0.9):
    plt.figure()
    plt.hist(y_prob, bins=40, color=COLORS["primary"], alpha=0.75, edgecolor="white")
    plt.axvspan(0, low, color=COLORS["secondary"], alpha=0.25, label=f"< {low}")
    plt.axvspan(high, 1, color=COLORS["tertiary"], alpha=0.25, label=f"> {high}")
    plt.xlabel("Predicted CT win probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bn_pred_prob_hist_extremes.png", dpi=200)
    plt.close()


def visualize_performance(y_true, y_prob, threshold=0.5):
    plot_roc(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    if not np.isclose(y_true.mean(), 0.5, atol=0.1):
        plot_pr(y_true, y_prob)
    plot_confusion(y_true, y_prob, threshold)
    plot_extremes(y_prob)

    print(f"AUC plot: {OUTPUT_DIR/'bn_roc_curve.png'}")
    print(f"Calibration plot: {OUTPUT_DIR/'bn_calibration_curve.png'}")
    print(f"Brier: {brier_score_loss(y_true, y_prob):.4f}, Logloss: {log_loss(y_true, y_prob):.4f}")
    if not np.isclose(y_true.mean(), 0.5, atol=0.1):
        print(f"PR plot: {OUTPUT_DIR/'bn_pr_curve.png'} (AP={average_precision_score(y_true, y_prob):.4f})")
    print(f"Confusion matrix (thr={threshold}): {OUTPUT_DIR/'bn_confusion_matrix.png'}")
    print(f"Extremes hist: {OUTPUT_DIR/'bn_pred_prob_hist_extremes.png'}")

def visualize_comparison():
    """Create visualization comparing BN vs LightGBM."""
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*80)
    
    # Load comparison data
    comparison_path = BN_DIR / 'bn_comparison.json'
    
    if not comparison_path.exists():
        print("Comparison data not found")
        print(f"  Expected: {comparison_path}")
        print("  Run bn_compare.py first")
        return
    
    with open(comparison_path) as f:
        comparison = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['accuracy', 'auc', 'logloss', 'brier']
    titles = ['Accuracy', 'AUC', 'Log Loss', 'Brier Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        bn_mean = comparison['bayesian_network'][metric]['mean']
        bn_std = comparison['bayesian_network'][metric]['std']
        lgbm_mean = comparison['lightgbm'][metric]['mean']
        lgbm_std = comparison['lightgbm'][metric]['std']
        
        # Bar plot with error bars
        models = ['Bayesian\nNetwork', 'LightGBM']
        values = [bn_mean, lgbm_mean]
        errors = [bn_std, lgbm_std]
        colors = [COLORS["primary"], COLORS["secondary"]]
        
        bars = ax.bar(
            models, values,
            yerr=errors,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=2,
            capsize=10,
            error_kw={'linewidth': 2}
        )
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{val:.4f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Add baseline for accuracy/AUC
        if metric in ['accuracy', 'auc']:
            ax.axhline(
                0.5, color=COLORS["accent"], linestyle='--',
                alpha=0.6, linewidth=2, label='Random Baseline'
            )
            ax.legend(fontsize=9)
        
        # Highlight winner
        winner = comparison['winner'][metric]
        if winner == 'BN':
            bars[0].set_edgecolor('gold')
            bars[0].set_linewidth(4)
        else:
            bars[1].set_edgecolor('gold')
            bars[1].set_linewidth(4)
    
    # Overall title
    fig.suptitle(
        'Bayesian Network vs LightGBM Performance Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(OUTPUT_DIR / 'bn_vs_lightgbm.png', dpi=300, bbox_inches='tight')
    print("Saved bn_vs_lightgbm.png")
    plt.close()


def create_inference_example_viz():
    """Visualize inference examples."""
    print("\n" + "="*80)
    print("CREATING INFERENCE EXAMPLES VISUALIZATION")
    print("="*80)
    
    # Load inference examples
    examples_path = BN_DIR / 'bn_inference_examples.json'
    
    if not examples_path.exists():
        print("Inference examples not found")
        return
    
    with open(examples_path) as f:
        examples = json.load(f)

    scenario_names = list(examples.keys())
    if len(scenario_names) > 5:
        rng = np.random.default_rng(42)
        scenario_names = list(rng.choice(scenario_names, size=5, replace=False))

    def pretty_label(ev):
        import textwrap
        label = ", ".join(f"{v}" for v in ev.values())
        return "\n".join(textwrap.wrap(label, width=20))

    labels = [pretty_label(examples[name]['evidence']) for name in scenario_names]
    ct_probs = [examples[name]['ct_win_prob'] for name in scenario_names]
    t_probs = [examples[name]['t_win_prob'] for name in scenario_names]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(labels))
    
    # Stacked horizontal bar
    ax.barh(y_pos, ct_probs, alpha=0.85, label='CT Win', color=COLORS["primary"])
    ax.barh(y_pos, t_probs, left=ct_probs, alpha=0.85, label='T Win', color=COLORS["secondary"])
    
    # Add probability labels
    for i, (ct_prob, t_prob) in enumerate(zip(ct_probs, t_probs)):
        # CT label
        if ct_prob > 0.1:
            ax.text(ct_prob/2, i, f'{ct_prob:.1%}',
                   ha='center', va='center', fontweight='bold', fontsize=10)
        # T label
        if t_prob > 0.1:
            ax.text(ct_prob + t_prob/2, i, f'{t_prob:.1%}',
                   ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Win Probability', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Network Inference Examples\n(Scenario-Based Predictions)',
                fontsize=14, fontweight='bold')
    ax.axvline(0.5, color=COLORS["accent"], linestyle='--', alpha=0.5, linewidth=2)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bn_inference_examples.png', dpi=300, bbox_inches='tight')
    print("Saved bn_inference_examples.png")
    plt.close()


def main():
    print("="*80)
    print("BAYESIAN NETWORK VISUALIZATION")
    print("="*80)
    
    # Load model
    model, features, structure, oof_true, oof_pred = load_model()
    
    # Create visualizations
    visualize_network_structure(model)
    visualize_performance(oof_true, oof_pred)
    visualize_comparison()
    create_inference_example_viz()
    
if __name__ == "__main__":
    main()
