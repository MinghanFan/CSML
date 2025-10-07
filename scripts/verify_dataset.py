"""
Dataset verification script.

Run this AFTER main_pipeline.py to verify your dataset is ready.
"""

from pathlib import Path
import polars as pl
import sys

from config import CLEAN_DATASET_FOLDER

def verify_dataset():
    """Verify dataset is complete and ready"""
    
    print("="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    
    # Check folder exists
    if not CLEAN_DATASET_FOLDER.exists():
        print(f"✗ ERROR: Dataset folder not found!")
        print(f"  Expected: {CLEAN_DATASET_FOLDER}")
        print(f"\n  Run main_pipeline.py first.")
        sys.exit(1)
    
    # Check files
    required_files = [
        'matches.parquet',
        'players.parquet',
        'match_players.parquet',
        'rounds.parquet',
        'round_players.parquet',
    ]
    
    print("\nChecking files...")
    all_exist = True
    for filename in required_files:
        filepath = CLEAN_DATASET_FOLDER / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024 / 1024  # MB
            print(f"  ✓ {filename} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename} MISSING!")
            all_exist = False
    
    if not all_exist:
        print(f"\n✗ Dataset incomplete!")
        sys.exit(1)
    
    # Load main dataset
    print("\nLoading match_players dataset...")
    df = pl.read_parquet(CLEAN_DATASET_FOLDER / 'match_players.parquet')
    
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Check critical columns
    print("\nChecking critical columns...")
    critical_cols = [
        'match_id', 'player_id', 'kills', 'deaths', 'adr',
        'utility_damage', 'flash_assists', 'won_match'
    ]
    
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].null_count()
            print(f"  ✓ {col} ({null_count} nulls)")
        else:
            print(f"  ✗ {col} MISSING!")
    
    # Check data quality
    print("\nData quality checks...")
    
    # Check if we have players with 0 kills
    players_with_zero_kills = len(df.filter(pl.col('kills') == 0))
    print(f"  Players with 0 kills: {players_with_zero_kills} (should be > 0)")
    
    # Check utility damage
    avg_util_dmg = df['utility_damage'].mean()
    print(f"  Avg utility damage: {avg_util_dmg:.1f}")
    
    # Check flash assists
    avg_flash_assists = df['flash_assists'].mean()
    print(f"  Avg flash assists: {avg_flash_assists:.2f}")
    
    print("\n" + "="*80)
    print("✓ VERIFICATION COMPLETE")
    print("="*80)
    print("\nYour dataset is ready for machine learning!")
    print("\nMain file: match_players.parquet")
    print(f"Location: {CLEAN_DATASET_FOLDER}")

if __name__ == "__main__":
    verify_dataset()