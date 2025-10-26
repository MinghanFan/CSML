"""
Player style clustering
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


DEFAULT_FEATURES: List[str] = [
    "kills_per_round",
    "deaths_per_round",
    "adr",
    "first_kill_rate",
    "survival_rate_ratio",
    "multi_kill_rate",
    "headshot_rate",
    "smokes_per_round",
    "flashes_per_round",
    "utility_damage_per_round",
    "flash_assists_per_round",
    "clutch_success_rate",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rounds_survived = df["rounds_survived"].fillna(0)
    deaths = df["deaths"].fillna(0)
    rounds_played = (rounds_survived + deaths).replace(0, 1)

    df["kills_per_round"] = df["kills"].fillna(0) / rounds_played
    df["deaths_per_round"] = deaths / rounds_played
    df["first_kill_rate"] = df["first_kills"].fillna(0) / rounds_played
    df["multi_kill_rate"] = df["multi_kill_rounds"].fillna(0) / rounds_played
    df["smokes_per_round"] = df["smokes_thrown"].fillna(0) / rounds_played
    df["flashes_per_round"] = df["flashes_thrown"].fillna(0) / rounds_played
    df["utility_damage_per_round"] = df["utility_damage"].fillna(0) / rounds_played
    df["flash_assists_per_round"] = df["flash_assists"].fillna(0) / rounds_played

    clutches_attempted = df["clutches_attempted"].replace(0, np.nan)
    df["clutch_success_rate"] = (df["clutches_won"] / clutches_attempted).fillna(0)

    df["survival_rate_ratio"] = df["survival_rate"].fillna(0) / 100.0
    df["headshot_rate"] = df["hsp"].fillna(0) / 100.0

    return df


def run_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    output_dir: Path,
) -> None:
    features = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["cluster"] = kmeans.fit_predict(X_pca)

    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = output_dir / "match_players_with_clusters.csv"
    df.to_csv(labeled_path, index=False)
    print(f"✓ Saved labeled dataset -> {labeled_path}")

    cluster_profile = df.groupby("cluster")[feature_cols].mean().sort_index()
    cluster_profile["win_rate"] = df.groupby("cluster")["won_match"].mean()

    profile_path = output_dir / "player_cluster_profiles.csv"
    cluster_profile.to_csv(profile_path)
    print(f"✓ Saved cluster profiles -> {profile_path}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df["cluster"],
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Player Style Clusters (PCA)")
    plot_path = output_dir / "player_clusters_pca.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✓ Saved PCA scatter plot -> {plot_path}")

    print("\nCluster win rates:")
    print(cluster_profile["win_rate"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Player style clustering pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("clean_dataset/match_players.csv"),
        help="Path to match_players CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clean_dataset"),
        help="Directory to save clustering outputs.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="Number of KMeans clusters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} match-player rows")

    df = engineer_features(df)
    feature_cols = [col for col in DEFAULT_FEATURES if col in df.columns]

    print("Using feature columns:", feature_cols)
    run_clustering(df, feature_cols, args.clusters, args.output_dir)


if __name__ == "__main__":
    main()
