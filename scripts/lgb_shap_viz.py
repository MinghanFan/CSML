"""
Create SHAP summary plots for the LightGBM model.
"""


from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap

from lgb_model import build_features, load_data

MODEL_PATH = Path("lgb_analysis/lgb_model.txt")
ARTIFACT_PATH = Path("lgb_analysis/lgb_model.pkl")
OUTPUT_DIR = Path("lgb_analysis")


def main(sample_size: int = 40000):
    artifacts = joblib.load(ARTIFACT_PATH)
    feature_cols = artifacts["feature_columns"]
    min_round: int = artifacts["min_round"]

    # plt.rcParams.update({"font.size": 6})

    rounds, players, matches = load_data()
    df = build_features(rounds, players, matches)
    df = df[df["round_num"] >= min_round].reset_index(drop=True)
    X = df[feature_cols].fillna(0.0).astype(np.float32)

    model = lgb.Booster(model_file=str(MODEL_PATH))

    # Sample for faster plotting
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # positive class
    shap_values = np.asarray(shap_values, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="violin",
        plot_size=0.2,
        show=False,
    )
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=6)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6)
        ax.set_ylabel(ax.get_ylabel(), fontsize=6)
        for label in ax.get_yticklabels():
            label.set_fontsize(8)
        for label in ax.get_xticklabels():
            label.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_violin.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
