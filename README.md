# CSML

## Remote Demo Processing

Large demo collections can now be processed without storing the `.dem` files
on disk. Use `scripts/remote_pipeline.py` to download each file into a
temporary workspace, parse it, and immediately feed it into the extractor.

Example runs:

```bash
# Discover .dem files via HTTP directory listings
python scripts/remote_pipeline.py --root-url https://example.com/demos/

# Or provide an explicit manifest with one URL per line
python scripts/remote_pipeline.py --manifest demo_manifest.txt

# Or point to a local folder (e.g. external drive) containing .dem files
python scripts/remote_pipeline.py --local-folder "/Volumes/TOSHIBA EXT/Demo_2025"

# Note: Files starting with `._` (macOS metadata) are automatically ignored.
```

The script keeps track of processed URLs in
`clean_dataset/processed_remote_demos.txt`, so re-running it will resume where
it left off. Pass `--no-resume` to start from scratch. The aggregated parquet
and CSV outputs are written to `clean_dataset/` just like the local pipeline.

Requires `requests` in addition to the existing AWPy/Polars dependencies.

## Performance Score

`match_players.performance_score` is now a per-round normalized composite that blends
impact, survival, utility, and clutch contributions. This keeps scores comparable
across matches of different lengths while rewarding high-impact play.

## Player Style Clustering

Run the full clustering workflow (feature engineering → PCA → KMeans) with:

```bash
python scripts/cluster.py --clusters 6
```

Outputs land in `clean_dataset/`:

- `match_players_with_clusters.csv` – original rows plus `cluster` label
- `player_cluster_profiles.csv` – feature means + win rate per cluster
- `player_clusters_pca.png` – PCA scatter plot for the paper

### Choosing the Right Number of Clusters

Use the evaluation helper to sweep through multiple K values, inspect inertia,
silhouette scores, and cluster profiles:

```bash
python scripts/profile.py --k-min 3 --k-max 8
```

Artifacts land in `clean_dataset/cluster_eval/` (metrics CSV, elbow/silhouette
plots, and per-K profile tables) so you can justify the final cluster count.
