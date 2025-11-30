"""
Creates visualizations for the Bayesian Network.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration
BN_DIR = Path("bn_analysis")
OUTPUT_DIR = Path("bn_analysis")

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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
    
    print(f"Loaded BN model")
    print(f"  Nodes: {len(model.nodes())}")
    print(f"  Edges: {len(model.edges())}")
    print(f"  Features: {len(features)}")
    
    return model, features, structure


def visualize_network_structure(model):
    """Visualize the Bayesian Network structure."""
    print("\n" + "="*80)
    print("CREATING NETWORK STRUCTURE VISUALIZATION")
    print("="*80)
    
    try:
        import networkx as nx
    except ImportError:
        print("Networkx not installed, skipping structure visualization")
        print("Install with: pip install networkx --break-system-packages")
        return
    
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
            node_colors.append('#FF6B6B')  # Red
            node_sizes.append(4000)
        elif node in ['equip_advantage', 'momentum', 'recent_performance']:
            node_colors.append('#4ECDC4')  # Teal (primary predictors)
            node_sizes.append(3500)
        else:
            node_colors.append('#95E1D3')  # Light teal (supporting features)
            node_sizes.append(3000)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='black',
        linewidths=2.5,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#2C3E50',
        arrows=True,
        arrowsize=25,
        arrowstyle='->',
        width=2.5,
        connectionstyle='arc3,rad=0.1',
        ax=ax
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
        Patch(facecolor='#FF6B6B', edgecolor='black', linewidth=2, label='Outcome (Target)'),
        Patch(facecolor='#4ECDC4', edgecolor='black', linewidth=2, label='Primary Predictors'),
        Patch(facecolor='#95E1D3', edgecolor='black', linewidth=2, label='Supporting Features')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Add note
    note_text = (
        "Edges represent causal relationships:\n"
        "- Equipment/Momentum/Performance → Outcome (direct predictors)\n"
        "- Performance → Momentum (psychological effect)\n"
        "- Round Phase → Buy Phase → Equipment (economic cycles)"
    )
    ax.text(
        0.02, 0.02,
        note_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bn_structure.png', dpi=300, bbox_inches='tight')
    print("Saved bn_structure.png")
    plt.close()


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
        colors = ['#4ECDC4', '#FF6B6B']
        
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
                0.5, color='red', linestyle='--',
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
    ax.barh(y_pos, ct_probs, alpha=0.8, label='CT Win', color='#4ECDC4')
    ax.barh(y_pos, t_probs, left=ct_probs, alpha=0.8, label='T Win', color='#FF6B6B')
    
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
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bn_inference_examples.png', dpi=300, bbox_inches='tight')
    print("Saved bn_inference_examples.png")
    plt.close()


def main():
    """Main execution."""
    
    print("="*80)
    print("BAYESIAN NETWORK VISUALIZATION")
    print("="*80)
    
    # Load model
    model, features, structure = load_model()
    
    # Create visualizations
    visualize_network_structure(model)
    visualize_comparison()
    create_inference_example_viz()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  - bn_structure.png - Network diagram")
    print(f"  - bn_vs_lightgbm.png - Performance comparison")
    print(f"  - bn_inference_examples.png - Scenario predictions")


if __name__ == "__main__":
    main()
