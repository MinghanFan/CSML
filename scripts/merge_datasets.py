"""
Utility script to merge an existing clean dataset with a new batch that was
produced via the remote pipeline (or any other run with custom prefixes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import polars as pl

DEFAULT_TABLES: List[str] = [
    "matches",
    "players",
    "match_players",
    "rounds",
    "round_players",
]


def normalize_prefix(prefix: str) -> str:
    trimmed = prefix.strip()
    if not trimmed:
        return ""
    if trimmed.endswith(("_", "-", ".")):
        return trimmed
    return f"{trimmed}_"


def load_tables(folder: Path, prefix: str, tables: Iterable[str]) -> Dict[str, pl.DataFrame]:
    dataset: Dict[str, pl.DataFrame] = {}
    for table in tables:
        parquet_path = folder / f"{prefix}{table}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing table: {parquet_path}")
        dataset[table] = pl.read_parquet(parquet_path)
    return dataset


def write_tables(
    tables: Dict[str, pl.DataFrame],
    output_folder: Path,
    output_prefix: str,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        filename = f"{output_prefix}{name}" if output_prefix else name
        parquet_path = output_folder / f"{filename}.parquet"
        csv_path = output_folder / f"{filename}.csv"
        df.write_parquet(parquet_path)
        df.write_csv(csv_path)
        print(f"  saved {filename}: {df.height} rows ->")
        print(f"    {parquet_path}")
        print(f"    {csv_path}")


def merge_players_table(base_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
    combined = pl.concat([base_df, new_df], how="vertical_relaxed")

    if combined.is_empty():
        return combined.with_columns(
            pl.col("total_matches").cast(pl.Int32),
            pl.col("avg_rank").cast(pl.Float32),
        )

    combined = combined.with_columns(
        pl.col("total_matches").cast(pl.Int64),
        pl.col("avg_rank").cast(pl.Float64),
        (pl.col("avg_rank") * pl.col("total_matches")).alias("weighted_rank"),
    )

    aggregated = (
        combined.group_by("player_id")
        .agg(
            [
                pl.col("total_matches").sum().alias("total_matches"),
                pl.col("weighted_rank").sum().alias("weighted_rank_sum"),
                pl.col("player_name")
                .filter(pl.col("player_name").is_not_null() & (pl.col("player_name").str.len_chars() > 0))
                .first()
                .alias("player_name"),
                pl.col("primary_role")
                .filter(pl.col("primary_role").is_not_null() & (pl.col("primary_role") != "unknown"))
                .first()
                .alias("primary_role"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("total_matches") > 0)
                .then(pl.col("weighted_rank_sum") / pl.col("total_matches"))
                .otherwise(0.0)
                .cast(pl.Float32)
                .alias("avg_rank"),
            ]
        )
        .drop("weighted_rank_sum")
        .with_columns(
            [
                pl.col("player_name").fill_null("").alias("player_name"),
                pl.col("primary_role").fill_null("unknown").alias("primary_role"),
                pl.col("total_matches").cast(pl.Int32),
            ]
        )
        .select(["player_id", "player_name", "total_matches", "primary_role", "avg_rank"])
    )

    return aggregated


def merge_datasets(
    *,
    base_folder: Path,
    base_prefix: str,
    new_folder: Path,
    new_prefix: str,
    output_folder: Path,
    output_prefix: str,
    tables: Iterable[str],
) -> None:
    print(f"\nLoading base dataset from: {base_folder} (prefix='{base_prefix}')")
    base_tables = load_tables(base_folder, base_prefix, tables)

    print(f"Loading new dataset from: {new_folder} (prefix='{new_prefix}')")
    new_tables = load_tables(new_folder, new_prefix, tables)

    combined: Dict[str, pl.DataFrame] = {}
    print("\nCombining tables...")
    for name in tables:
        base_df = base_tables[name]
        new_df = new_tables[name]
        if name == "players":
            merged = merge_players_table(base_df, new_df)
        else:
            merged = pl.concat([base_df, new_df], how="vertical_relaxed")
        combined[name] = merged
        print(f"  {name}: base={base_df.height} + new={new_df.height} -> combined={merged.height}")

    print(f"\nWriting combined outputs to: {output_folder} (prefix='{output_prefix}')")
    write_tables(combined, output_folder, output_prefix)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a CSML clean dataset with another run that uses a custom output prefix/folder."
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        default=Path("clean_dataset"),
        help="Folder containing the existing clean dataset (default: clean_dataset).",
    )
    parser.add_argument(
        "--base-prefix",
        default="",
        help="Optional filename prefix for the base dataset (e.g., 'baseline_').",
    )
    parser.add_argument(
        "--new-folder",
        type=Path,
        required=True,
        help="Folder containing the new dataset to append.",
    )
    parser.add_argument(
        "--new-prefix",
        default="",
        help="Filename prefix used when creating the new dataset (e.g., 'remote_').",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("clean_dataset"),
        help="Destination folder for the combined dataset (default: clean_dataset).",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for the combined outputs (leave blank to overwrite the base names).",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help="List of table names to merge (default: matches players match_players rounds round_players).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_folder = args.base_folder.expanduser()
    new_folder = args.new_folder.expanduser()
    output_folder = args.output_folder.expanduser()
    base_prefix = normalize_prefix(args.base_prefix)
    new_prefix = normalize_prefix(args.new_prefix)
    output_prefix = normalize_prefix(args.output_prefix)

    merge_datasets(
        base_folder=base_folder,
        base_prefix=base_prefix,
        new_folder=new_folder,
        new_prefix=new_prefix,
        output_folder=output_folder,
        output_prefix=output_prefix,
        tables=args.tables,
    )


if __name__ == "__main__":
    main()
