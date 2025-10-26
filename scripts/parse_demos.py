"""
Demo parser script to parses raw .dem files.

Run before main_pipeline.py.
"""

from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count

from config import RAW_DEMOS_FOLDER, PARSED_DEMOS_FOLDER, NUM_WORKERS
from demo_utils import parse_demo_to_zip

def parse_single_demo(args):
    """Parse a single demo (for multiprocessing)"""
    demo_path, output_folder = args
    
    try:
        print(f"Parsing {demo_path.name}...")
        output_path = parse_demo_to_zip(demo_path, output_folder)
        
        print(f"  ✓ {demo_path.name} -> {output_path.name}")
        return {'success': True, 'demo': demo_path.name}
        
    except Exception as e:
        print(f"  ✗ {demo_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'demo': demo_path.name, 'error': str(e)}

def main():
    """Parse all demos"""
    
    print("="*80)
    print("CS2 DEMO PARSER")
    print("="*80)
    
    # Validate
    if not RAW_DEMOS_FOLDER.exists():
        print(f"✗ ERROR: Raw demos folder not found!")
        print(f"  Expected: {RAW_DEMOS_FOLDER}")
        print(f"\n  Create the folder and put your .dem files there.")
        sys.exit(1)
    
    demo_files = sorted(
        p for p in RAW_DEMOS_FOLDER.glob("*.dem")
        if not p.name.startswith("._")
    )
    if len(demo_files) == 0:
        print(f"✗ ERROR: No .dem files found!")
        print(f"  Folder: {RAW_DEMOS_FOLDER}")
        print(f"\n  Copy your demo files to this folder first.")
        sys.exit(1)
    
    print(f"\nFound {len(demo_files)} demos to parse")
    PARSED_DEMOS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Determine workers
    num_workers = NUM_WORKERS if NUM_WORKERS else max(1, cpu_count() - 1)
    print(f"Using {num_workers} parallel workers\n")
    
    # Prepare args
    parse_args = [(demo_path, PARSED_DEMOS_FOLDER) for demo_path in demo_files]
    
    # Parse
    if num_workers == 1:
        # Serial processing
        results = [parse_single_demo(args) for args in parse_args]
    else:
        # Parallel processing
        with Pool(num_workers) as pool:
            results = pool.map(parse_single_demo, parse_args)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"\n{'='*80}")
    print("PARSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful: {successful}/{len(demo_files)}")
    print(f"✗ Failed: {failed}/{len(demo_files)}")
    
    if failed > 0:
        print(f"\nFailed demos:")
        for r in results:
            if not r['success']:
                print(f"  - {r['demo']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nParsed demos saved to: {PARSED_DEMOS_FOLDER}")
    
    if successful > 0:
        print(f"\n✓ Ready for next step!")
        print(f"  Run: python scripts/main_pipeline.py")
    else:
        print(f"\n✗ No demos parsed successfully. Check errors above.")

if __name__ == "__main__":
    main()
