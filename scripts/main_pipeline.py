"""
Main pipeline script - RUN THIS FILE!

This orchestrates the entire data acquisition process.
"""

from pathlib import Path
import sys

# Import our modules
from config import (
    RAW_DEMOS_FOLDER,
    PARSED_DEMOS_FOLDER, 
    CLEAN_DATASET_FOLDER,
    VALID_MAPS,
    NUM_WORKERS
)
from data_extractor import CS2DataExtractor

def main():
    """
    Main pipeline execution.
    
    Steps:
    1. Validate input folders
    2. Extract data from parsed demos
    3. Create clean DataFrames
    4. Save results
    5. Generate summary
    """
    
    print("="*80)
    print("CS2 DATA ACQUISITION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Parsed demos folder: {PARSED_DEMOS_FOLDER}")
    print(f"  Output folder: {CLEAN_DATASET_FOLDER}")
    
    # Step 1: Validate
    print(f"\n{'='*80}")
    print("STEP 1: VALIDATION")
    print(f"{'='*80}")
    
    if not PARSED_DEMOS_FOLDER.exists():
        print(f"✗ ERROR: Parsed demos folder not found!")
        print(f"  Expected: {PARSED_DEMOS_FOLDER}")
        print(f"\n  You need to parse your demos first.")
        print(f"  Run: python scripts/parse_demos.py")
        sys.exit(1)
    
    demo_zips = list(PARSED_DEMOS_FOLDER.glob("*.zip"))
    if len(demo_zips) == 0:
        print(f"✗ ERROR: No parsed demos found!")
        print(f"  Folder: {PARSED_DEMOS_FOLDER}")
        print(f"\n  You need to parse your demos first.")
        print(f"  Run: python scripts/parse_demos.py")
        sys.exit(1)
    
    print(f"✓ Found {len(demo_zips)} parsed demos")
    
    # Step 2: Extract
    print(f"\n{'='*80}")
    print("STEP 2: DATA EXTRACTION")
    print(f"{'='*80}")
    
    extractor = CS2DataExtractor(
        parsed_demos_folder=PARSED_DEMOS_FOLDER,
        output_folder=CLEAN_DATASET_FOLDER
    )
    
    # Process all demos
    results = extractor.process_all_demos()
    
    successful = sum(1 for r in results if r['success'])
    if successful == 0:
        print(f"\n✗ ERROR: No demos processed successfully!")
        sys.exit(1)
    
    # Step 3: Create DataFrames
    print(f"\n{'='*80}")
    print("STEP 3: CREATE DATAFRAMES")
    print(f"{'='*80}")
    
    dataframes = extractor.create_dataframes()
    
    # Step 4: Save
    print(f"\n{'='*80}")
    print("STEP 4: SAVE DATA")
    print(f"{'='*80}")
    
    extractor.save_dataframes(dataframes)
    
    # Step 5: Summary
    print(f"\n{'='*80}")
    print("STEP 5: GENERATE SUMMARY")
    print(f"{'='*80}")
    
    summary = extractor.generate_summary(dataframes)
    extractor.save_summary(summary)
    
    # Done!
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"\nYour clean dataset is ready at:")
    print(f"  {CLEAN_DATASET_FOLDER}")
    print(f"\nMain file for ML:")
    print(f"  {CLEAN_DATASET_FOLDER / 'match_players.parquet'}")
    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/verify_dataset.py")
    print(f"  2. Start feature engineering and ML model training")

if __name__ == "__main__":
    main()
