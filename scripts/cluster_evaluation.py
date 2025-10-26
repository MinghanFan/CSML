"""
Cluster evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cluster import DEFAULT_FEATURES, engineer_features

matplotlib.use("Agg")


def evaluate_clusters(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_values: List[int],
    random_state: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Run PCA + KMeans for each K and capture metrics/profiles."""
    features = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    results: List[Dict[str, float]] = []
    profile_dir = output_dir / "cluster_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        print(f"Evaluating k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(X_pca)

        df[f"cluster_k{k}"] = labels

        inertia = kmeans.inertia_
        silhouette = (
            silhouette_score(X_pca, labels) if k > 1 else float("nan")
        )

        counts = np.bincount(labels)
        min_size = counts.min()
        max_size = counts.max()

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette,
                "min_cluster_size": min_size,
                "max_cluster_size": max_size,
            }
        )

        cluster_profile = df.groupby(f"cluster_k{k}")[feature_cols].mean()
        cluster_profile["win_rate"] = (
            df.groupby(f"cluster_k{k}")["won_match"].mean()
        )

        profile_path = profile_dir / f"cluster_profiles_k{k}.csv"
        cluster_profile.to_csv(profile_path)
        print(f"  ✓ Saved profiles -> {profile_path}")

    metrics_df = pd.DataFrame(results)
    metrics_path = output_dir / "cluster_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Saved metrics summary -> {metrics_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    plt.title("KMeans Inertia vs K")
    plt.xlabel("K")
    plt.ylabel("Inertia (WCSS)")
    plt.tight_layout()
    plt.savefig(output_dir / "inertia_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_vs_k.png", dpi=200)
    plt.close()

    print("✓ Saved elbow and silhouette plots")

    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate different cluster counts for player styles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("clean_dataset/match_players.csv"),
        help="Path to match_players CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clean_dataset/cluster_eval"),
        help="Directory to store evaluation artifacts.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum number of clusters to evaluate.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=8,
        help="Maximum number of clusters to evaluate.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    df = engineer_features(df)
    feature_cols = [col for col in DEFAULT_FEATURES if col in df.columns]
    print("Using feature columns:", feature_cols)

    k_values = list(range(args.k_min, args.k_max + 1))
    metrics_df = evaluate_clusters(
        df=df,
        feature_cols=feature_cols,
        k_values=k_values,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print("\nCluster evaluation summary:")
    print(metrics_df)


if __name__ == "__main__":
    main()
