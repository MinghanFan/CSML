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
`<output-folder>/processed_remote_demos.txt`, so re-running it will resume
where it left off. Pass `--no-resume` to start from scratch.

Use `--output-folder` to send the generated CSV/Parquet files to a separate
directory (e.g., `clean_dataset/remote_run/`) and add `--output-prefix` to
prepend a label like `remote_` to each filename. Both options make it easy to
process fresh demos without overwriting your existing dataset and merge the
results later on.

Progress is still tracked in `clean_dataset/processed_remote_demos.txt` by
default. Pass `--progress-file "<output-folder>/processed_remote_demos.txt"`
if you prefer to keep separate logs per run.

To merge a prefixed batch back into the main dataset, use
`scripts/merge_datasets.py`:

```bash
python scripts/merge_datasets.py \
  --base-folder clean_dataset \
  --new-folder clean_dataset/remote_run \
  --new-prefix remote_ \
  --output-folder clean_dataset \
  --output-prefix combined_
```

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
