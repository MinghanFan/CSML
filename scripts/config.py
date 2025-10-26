"""
Configuration file for CS2 data acquisition pipeline.
"""

from pathlib import Path

# === PROJECT FOLDERS ===
PROJECT_ROOT = Path(__file__).parent.parent

RAW_DEMOS_FOLDER = PROJECT_ROOT / "raw_demos"

PARSED_DEMOS_FOLDER = PROJECT_ROOT / "parsed_demos"

CLEAN_DATASET_FOLDER = PROJECT_ROOT / "clean_dataset"

# Maps to include (empty list = all maps)
VALID_MAPS = [
    "de_dust2",
    "de_mirage", 
    "de_inferno",
    "de_nuke",
    "de_overpass",
    "de_vertigo",
    "de_ancient",
    "de_anubis",
    "de_anubis"
]

# === PARSING SETTINGS ===
# Number of parallel workers for parsing (None = auto-detect CPU count - 1)
NUM_WORKERS = None

print(f"✓ Configuration loaded")
print(f"  Raw demos: {RAW_DEMOS_FOLDER}")
print(f"  Output: {CLEAN_DATASET_FOLDER}")
