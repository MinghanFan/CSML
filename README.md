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
