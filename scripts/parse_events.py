"""
Slim awpy event parser: parse demo files into per-match event parquet files,
"""

from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
from typing import List, Dict

import pandas as pd
import polars as pl
from awpy.demo import Demo

CLEAN_ROOT = Path("clean_dataset")
PER_MATCH_ROOT = CLEAN_ROOT / "_per_match"
SOURCE_MARKER = "source_path.txt"

EVENT_FILES = [
    "kills.parquet",
    "damages.parquet",
    "grenades.parquet",
    "bomb.parquet",
    "shots.parquet",
    "infernos.parquet",
    "smokes.parquet",
    "footsteps.parquet",
    "ticks.parquet",
]


def derive_match_id(demo_path: Path) -> str:
    return demo_path.stem  # keep consistent with your existing pipeline


def ensure_dir(match_id: str, source_path: Path) -> Path:
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    out_dir = PER_MATCH_ROOT / f"{match_id}__{source_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def side_to_upper(x):
    if x is None:
        return None
    s = str(x).lower()
    if s == "t":
        return "T"
    if s == "ct":
        return "CT"
    return str(x).upper()


def already_parsed(out_dir: Path, source_path: Path) -> bool:
    if not all((out_dir / f).exists() for f in EVENT_FILES):
        return False
    marker_path = out_dir / SOURCE_MARKER
    if not marker_path.exists():
        return False
    recorded = marker_path.read_text(encoding="utf-8").strip()
    return recorded == str(source_path)


def write_event(df_pd: pd.DataFrame, cols, out_path: Path, side_cols=None, match_id=None):
    if df_pd is None or df_pd.empty:
        # Write empty schema for consistency
        schema = {}
        for c in cols:
            if c in ("round_num", "tick"):
                schema[c] = pl.Int64
            else:
                schema[c] = pl.Utf8
        pl.DataFrame(schema=schema).write_parquet(out_path)
        return

    keep = [c for c in cols if c in df_pd.columns]
    df = df_pd[keep].copy()

    if side_cols:
        for sc in side_cols:
            if sc in df.columns:
                df[sc] = df[sc].map(lambda v: side_to_upper(v) if pd.notna(v) else pd.NA)

    if "match_id" not in df.columns and match_id is not None:
        df["match_id"] = match_id

    pl.from_pandas(df).write_parquet(out_path)


def to_pandas_safe(component) -> pd.DataFrame:
    if component is None:
        return pd.DataFrame()
    try:
        return component.to_pandas(copy=True)
    except TypeError:
        # Older/newer awpy may not support copy keyword
        return component.to_pandas()


def parse_one(demo_path: Path) -> None:
    match_id = derive_match_id(demo_path)
    source_path = demo_path.resolve()
    out_dir = ensure_dir(match_id, source_path)

    if already_parsed(out_dir, source_path):
        print(f"[skip] {source_path} -> cached events")
        return

    print(f"[parse] {demo_path} -> {out_dir}")

    dem = Demo(str(demo_path))
    dem.parse()

    # Kills
    kills_cols = ["match_id", "round_num", "tick", "attacker_steamid", "victim_steamid",
                  "attacker_side", "victim_side", "weapon", "hitgroup"]
    df = to_pandas_safe(dem.kills) if hasattr(dem, "kills") else pd.DataFrame()
    write_event(df, kills_cols, out_dir / "kills.parquet", side_cols=["attacker_side", "victim_side"], match_id=match_id)

    # Damages
    dmg_cols = ["match_id", "round_num", "attacker_steamid", "victim_steamid",
                "attacker_side", "victim_side", "dmg_health", "weapon"]
    df = to_pandas_safe(dem.damages) if hasattr(dem, "damages") else pd.DataFrame()
    write_event(df, dmg_cols, out_dir / "damages.parquet", side_cols=["attacker_side", "victim_side"], match_id=match_id)

    # Grenades
    nade_cols = ["match_id", "round_num", "thrower_steamid", "thrower_side", "nade_type", "tick"]
    df = to_pandas_safe(dem.grenades) if hasattr(dem, "grenades") else pd.DataFrame()
    write_event(df, nade_cols, out_dir / "grenades.parquet", side_cols=["thrower_side"], match_id=match_id)

    # Bomb
    bomb_cols = ["match_id", "round_num", "event", "bombsite", "tick"]
    df = to_pandas_safe(dem.bomb) if hasattr(dem, "bomb") else pd.DataFrame()
    write_event(df, bomb_cols, out_dir / "bomb.parquet", match_id=match_id)

    # Shots
    shot_cols = ["match_id", "round_num", "tick", "shooter_steamid", "shooter_side", "weapon"]
    df = to_pandas_safe(dem.shots) if hasattr(dem, "shots") else pd.DataFrame()
    write_event(df, shot_cols, out_dir / "shots.parquet", side_cols=["shooter_side"], match_id=match_id)

    # Infernos
    inf_cols = ["match_id", "round_num", "tick", "thrower_steamid", "thrower_side"]
    df = to_pandas_safe(dem.infernos) if hasattr(dem, "infernos") else pd.DataFrame()
    write_event(df, inf_cols, out_dir / "infernos.parquet", side_cols=["thrower_side"], match_id=match_id)

    # Smokes
    smo_cols = ["match_id", "round_num", "tick", "thrower_steamid", "thrower_side"]
    df = to_pandas_safe(dem.smokes) if hasattr(dem, "smokes") else pd.DataFrame()
    write_event(df, smo_cols, out_dir / "smokes.parquet", side_cols=["thrower_side"], match_id=match_id)

    # Footsteps
    foot_cols = ["match_id", "round_num", "tick", "player_steamid", "player_side"]
    df = to_pandas_safe(dem.footsteps) if hasattr(dem, "footsteps") else pd.DataFrame()
    write_event(df, foot_cols, out_dir / "footsteps.parquet", side_cols=["player_side"], match_id=match_id)

    # Ticks
    tick_cols = ["match_id", "round_num", "tick"]
    df = to_pandas_safe(dem.ticks) if hasattr(dem, "ticks") else pd.DataFrame()
    write_event(df, tick_cols, out_dir / "ticks.parquet", match_id=match_id)
    (out_dir / SOURCE_MARKER).write_text(str(source_path), encoding="utf-8")

    print(f"[done] {match_id}")


def concat_parquet(paths: List[Path], out_path: Path) -> None:
    frames = []
    for p in paths:
        try:
            frames.append(pl.read_parquet(p))
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}")
    if frames:
        df = pl.concat(frames, how="vertical_relaxed")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        print(f"[combine] {out_path} ({df.height} rows)")
    else:
        print(f"[combine] No inputs for {out_path.name}")


def combine_all() -> None:
    base = PER_MATCH_ROOT
    if not base.exists():
        print("[combine] Nothing to combine; per-match folder missing.")
        return

    for fname in EVENT_FILES:
        paths = sorted(base.glob(f"*/{fname}"))
        concat_parquet(paths, CLEAN_ROOT / fname)


def find_demos(root: Path) -> List[Path]:
    demos = []
    for pat in ("*.dem", "*.DEM"):
        demos.extend(root.rglob(pat))
    return sorted(demos)


def main():
    ap = argparse.ArgumentParser(description="Slim awpy event parser (no rounds/round_players)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Parse demos -> per-match event parquet files")
    p_parse.add_argument("--in", dest="in_dir", type=Path, required=True, help="Folder with .dem files (recursive)")

    sub.add_parser("combine", help="Combine per-match event files -> clean_dataset/*.parquet")

    args = ap.parse_args()

    if args.cmd == "parse":
        demos = find_demos(args.in_dir)
        if not demos:
            print(f"[parse] No .dem under {args.in_dir}")
            return
        for d in demos:
            try:
                parse_one(d)
            except Exception as e:
                print(f"[error] {d}: {e}")
    else:
        combine_all()


if __name__ == "__main__":
    main()
