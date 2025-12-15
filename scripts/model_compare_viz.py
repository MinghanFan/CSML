from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc

# --- CONFIGURATION ---
BASE_DIR = Path(".") 
BN_DIR = BASE_DIR / "bn_analysis"
LGB_DIR = BASE_DIR / "lgb_analysis"
BASE_RES_DIR = BASE_DIR / "baseline_results"
OUTPUT_DIR = BASE_DIR / "model_comparison"
PALETTE = sns.color_palette("tab10")

# ---------------------
FILES = {
    "Baseline": {
        "metrics": BASE_RES_DIR / "baseline_metrics.json",
        "model": None, 
        "color": 7 # Grey 
    },
    "Bayes": {
        "metrics": BN_DIR / "bn_metrics.json",
        "model": BN_DIR / "bn_model.pkl",
        "color": 0 # Blue
    },
    "LightGBM": {
        "metrics": LGB_DIR / "lgb_metrics.json",
        "model": LGB_DIR / "lgb_model.pkl",
        "color": 1 # Orange
    }
}



def load_metrics() -> dict:
    """Load pre-calculated metrics from JSON files for all models."""
    data = {}
    
    for name, paths in FILES.items():
        path = paths["metrics"]

        with open(path) as f:
            raw = json.load(f)
            
        # Extract mean/std for standard keys
        data[name] = {
            k: {"mean": raw["cv_mean"].get(k), "std": raw["cv_std"].get(k, 0)}
            for k in ["accuracy", "auc", "logloss", "brier"] 
            if k in raw["cv_mean"]
        }
    return data

def load_roc_data() -> dict:
    """Load predictions to build ROC curves."""
    curves = {}
    
    # Load Bayes & LightGBM
    for name in ["Bayes", "LightGBM"]:
        path = FILES[name]["model"]

        model_data = joblib.load(path)
        y_true = model_data["oof_true"]
        y_prob = model_data["oof_predictions"]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    # Construct Baseline ROC
    curves["Baseline"] = {
        "fpr": [0, 1], 
        "tpr": [0, 1], 
        "auc": 0.5
    }
        
    return curves

def plot_metrics(data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_map = {"Accuracy": "accuracy", "AUC": "auc", "Log Loss": "logloss", "Brier Score": "brier"}
    
    # Get common models present in data
    models = [m for m in FILES.keys() if m in data]
    
    for ax, (title, key) in zip(axes.flatten(), metrics_map.items()):
        vals = [data[m][key]["mean"] for m in models]
        errs = [data[m][key]["std"] for m in models]
        colors = [PALETTE[FILES[m]["color"]] for m in models]
        
        bars = ax.bar(models, vals, yerr=errs, color=colors, alpha=0.8, capsize=5, edgecolor="black")
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Model Comparison Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_compare_metrics.png", dpi=300)
    plt.close()

def plot_roc(curves: dict):
    plt.figure(figsize=(7, 6))
    
    for name in FILES.keys():
        if name not in curves: continue
        
        c = curves[name]
        color = PALETTE[FILES[name]["color"]]
        
        # Diagonal line style for Baseline
        style = "--" if name == "Baseline" else "-"
        width = 1 if name == "Baseline" else 2
        
        plt.plot(c["fpr"], c["tpr"], color=color, linestyle=style, lw=width, 
                 label=f"{name} (AUC {c['auc']:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_compare_roc.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics...")
    metrics_data = load_metrics()
    plot_metrics(metrics_data)
    print(f"Saved metrics plot to {OUTPUT_DIR}")

    print("Loading ROC data...")
    roc_data = load_roc_data()
    plot_roc(roc_data)
    print(f"Saved ROC plot to {OUTPUT_DIR}")