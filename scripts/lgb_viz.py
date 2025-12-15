"""
Generate visualizations for LightGBM model evaluation.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve
)

ARTIFACT_PATH = Path("lgb_analysis/lgb_model.pkl")
OUTPUT_DIR = Path("lgb_analysis")

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


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color=COLORS["primary"], label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=0.8, color=COLORS["grey"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
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
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=200)
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
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

def plot_extremes(y_prob, low=0.1, high=0.9):
    # Histogram with shaded extreme regions
    plt.figure()
    plt.hist(y_prob, bins=40, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    plt.axvspan(0, low, color=COLORS["secondary"], alpha=0.25, label=f"< {low}")
    plt.axvspan(high, 1, color=COLORS["tertiary"], alpha=0.25, label=f"> {high}")
    plt.xlabel("Predicted CT win probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pred_prob_hist_extremes.png", dpi=200)
    plt.close()


def main(threshold: float = 0.5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    art = joblib.load(ARTIFACT_PATH)
    y_true = np.asarray(art["oof_true"])
    y_prob = np.asarray(art["oof_predictions"])

    plot_roc(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    if not np.isclose(y_true.mean(), 0.5, atol=0.1):
        plot_pr(y_true, y_prob)
    plot_confusion(y_true, y_prob, threshold)
    plot_extremes(y_prob)


if __name__ == "__main__":
    main()
