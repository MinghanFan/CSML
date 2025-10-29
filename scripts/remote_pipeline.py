"""
Remote demo processing pipeline.

Downloads demos on demand from a remote HTTP directory or a manifest file,
parses each demo into the AWPy parquet bundle, feeds the result directly into
`CS2DataExtractor`, and discards the temporary files immediately.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Union
from urllib.parse import urljoin, urlparse

import requests

from config import CLEAN_DATASET_FOLDER
from data_extractor import CS2DataExtractor
from demo_utils import parse_demo_to_zip

MB = 1024 * 1024


def normalize_prefix(prefix: str) -> str:
    trimmed = prefix.strip()
    if not trimmed:
        return ""
    if trimmed.endswith(("_", "-", ".")):
        return trimmed
    return f"{trimmed}_"


def normalize_directory_url(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def extract_links(html: str) -> Iterator[str]:
    href_pattern = re.compile(r'href=[\'"]?([^\'" >]+)')
    for match in href_pattern.finditer(html):
        yield match.group(1)


def walk_remote_tree(root_url: str) -> Iterator[str]:
    """
    Yield `.dem` file URLs beneath `root_url` by following directory listings.
    """
    root_url = normalize_directory_url(root_url)
    parsed_root = urlparse(root_url)
    allowed_netloc = parsed_root.netloc
    visited_dirs: Set[str] = set()
    pending_dirs: List[str] = [root_url]

    while pending_dirs:
        current = pending_dirs.pop()
        if current in visited_dirs:
            continue
        visited_dirs.add(current)

        response = requests.get(current, timeout=60)
        response.raise_for_status()

        for href in extract_links(response.text):
            if href.startswith("#"):
                continue
            absolute = urljoin(current, href)
            parsed = urlparse(absolute)
            if parsed.netloc != allowed_netloc:
                continue
            if not absolute.startswith(root_url):
                continue
            if absolute.endswith("/"):
                pending_dirs.append(absolute)
                continue
            if parsed.path.lower().endswith(".dem"):
                if Path(parsed.path).name.startswith("._"):
                    continue
                yield absolute


def load_manifest(manifest_path: Path) -> List[str]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def load_processed(progress_path: Path) -> Set[str]:
    if not progress_path.exists():
        return set()
    lines = progress_path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def append_processed(progress_path: Path, url: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(url + "\n")


def download_demo(url: str, destination: Path, chunk_size: int = 32 * MB) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    next_log_threshold = 200 * MB

    if total_size:
        print(f"    Downloading {total_size / MB:.1f} MB ...")
    else:
        print("    Downloading (size unknown) ...")

    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if not total_size and downloaded >= next_log_threshold:
                print(f"      {downloaded / MB:.1f} MB downloaded ...")
                next_log_threshold += 200 * MB

    if total_size:
        print(f"    Download complete ({downloaded / MB:.1f}/{total_size / MB:.1f} MB)")
    else:
        print(f"    Download complete ({downloaded / MB:.1f} MB)")


DemoSource = Union[str, Path]


def iter_demo_sources(
    root_url: Optional[str],
    manifest: Optional[Path],
    local_folders: Sequence[Path],
) -> Iterator[DemoSource]:
    if manifest:
        for url in load_manifest(manifest):
            yield url
    if root_url:
        for url in walk_remote_tree(root_url):
            yield url
    for folder in local_folders:
        expanded = folder.expanduser()
        if not expanded.exists():
            print(f"Warning: local folder does not exist: {expanded}", file=sys.stderr)
            continue
        for demo_path in expanded.rglob("*.dem"):
            if demo_path.is_file():
                if demo_path.name.startswith("._"):
                    continue
                yield demo_path


def process_remote_demos(
    demo_sources: Iterable[DemoSource],
    *,
    progress_file: Path,
    output_folder: Path,
    filename_prefix: str = "",
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    output_folder = output_folder.expanduser()
    progress_file = progress_file.expanduser()
    file_prefix = normalize_prefix(filename_prefix)

    print(f"\nOutputs will be written to: {output_folder}")
    if file_prefix:
        print(f"Filename prefix: {file_prefix}")

    if not resume and progress_file.exists():
        progress_file.unlink()
    processed_urls = load_processed(progress_file) if resume else set()
    skipped = 0
    successes = 0
    failures = 0

    with tempfile.TemporaryDirectory(prefix="parsed_cache_") as parsed_cache_dir:
        extractor = CS2DataExtractor(
            parsed_demos_folder=Path(parsed_cache_dir),
            output_folder=output_folder,
        )

        for idx, source in enumerate(demo_sources, start=1):
            if limit and successes >= limit:
                break
            source_id = (
                str(Path(source).resolve())
                if isinstance(source, Path)
                else source
            )
            if source_id in processed_urls:
                skipped += 1
                continue

            print(f"\n{'=' * 80}")
            print(f"[{idx}] Processing demo")
            print(source_id)

            try:
                with tempfile.TemporaryDirectory(prefix="demo_work_") as work_dir_str:
                    work_dir = Path(work_dir_str)
                    if isinstance(source, Path):
                        demo_path = source
                    else:
                        filename = Path(urlparse(source).path).name or f"demo_{idx}.dem"
                        local_demo_path = work_dir / filename
                        download_demo(source, local_demo_path)
                        demo_path = local_demo_path

                    if not demo_path.exists():
                        raise FileNotFoundError(f"Demo not found: {demo_path}")

                    parsed_zip = parse_demo_to_zip(demo_path, work_dir)
                    result = extractor.process_single_demo(parsed_zip)

                    if not result.get("success"):
                        raise RuntimeError(result.get("error", "Unknown processing error"))

                append_processed(progress_file, source_id)
                processed_urls.add(source_id)
                successes += 1
                print(f"  ✓ Completed {source_id}")

            except Exception as exc:
                failures += 1
                print(f"  ✗ Failed: {exc}")
                import traceback

                traceback.print_exc()

        if successes == 0:
            print("\nNo demos processed successfully. Nothing to save.")
            if failures:
                sys.exit(1)
            return

        print(f"\nSaving aggregated dataset ({successes} demos processed)...")
        dataframes = extractor.create_dataframes()
        extractor.save_dataframes(dataframes, prefix=file_prefix)
        summary = extractor.generate_summary(dataframes)
        summary_filename = f"{file_prefix}dataset_summary.json" if file_prefix else "dataset_summary.json"
        extractor.save_summary(summary, filename=summary_filename)

    print(f"\nDone. Successes: {successes}, Skipped: {skipped}, Failures: {failures}")
    if failures:
        sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process remote CS2 demos without storing them locally.")
    parser.add_argument("--root-url", help="HTTP root that exposes .dem files via directory listings.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a text file containing one demo URL per line."
    )
    parser.add_argument(
        "--local-folder",
        type=Path,
        action="append",
        default=[],
        help="Path to a folder containing .dem files (recursive). Can be used multiple times."
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=CLEAN_DATASET_FOLDER,
        help="Destination folder for generated CSV/Parquet outputs (defaults to CLEAN_DATASET_FOLDER)."
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix to prepend to output filenames (e.g., 'remote_')."
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=CLEAN_DATASET_FOLDER / "processed_remote_demos.txt",
        help="File used to track processed demo URLs for resumable runs (default: clean_dataset/processed_remote_demos.txt)."
    )
    parser.add_argument("--limit", type=int, help="Stop after processing N demos (useful for smoke tests).")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing progress file and process all demos again."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.root_url and not args.manifest and not args.local_folder:
        parser.error("You must provide --root-url, --manifest, --local-folder, or a combination.")

    output_folder = args.output_folder.expanduser()
    progress_file = args.progress_file.expanduser()

    try:
        demo_iter = iter_demo_sources(args.root_url, args.manifest, args.local_folder)
        process_remote_demos(
            demo_iter,
            progress_file=progress_file,
            output_folder=output_folder,
            filename_prefix=args.output_prefix,
            limit=args.limit,
            resume=not args.no_resume,
        )
    except requests.RequestException as exc:
        print(f"Network error while accessing remote demos: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
