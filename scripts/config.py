"""
Configuration file for CS2 data acquisition pipeline.
Edit the paths below to match your setup.
"""

from pathlib import Path

# === PROJECT FOLDERS ===
# Root folder of your project
PROJECT_ROOT = Path(__file__).parent.parent

# Where your raw .dem files are
RAW_DEMOS_FOLDER = PROJECT_ROOT / "raw_demos"

# Where parsed demos will be saved
PARSED_DEMOS_FOLDER = PROJECT_ROOT / "parsed_demos"

# Where final clean dataset will be saved
CLEAN_DATASET_FOLDER = PROJECT_ROOT / "clean_dataset"

# === DATA COLLECTION SETTINGS ===
# Minimum rounds per match (filter out short/surrendered matches)
MIN_ROUNDS = 16

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

# Tickrate (CS2 default is 64, competitive is 128)
TICKRATE = 128

print(f"✓ Configuration loaded")
print(f"  Raw demos: {RAW_DEMOS_FOLDER}")
print(f"  Output: {CLEAN_DATASET_FOLDER}")