"""
Shared utilities for parsing CS2 demo files.

This module centralizes the parsing configuration so different entry points
(`parse_demos.py`, `remote_pipeline.py`, etc.) stay in sync.
"""

from pathlib import Path
from typing import Iterable, Optional, Sequence

import polars as pl
from awpy import Demo

# Default parsing configuration
DEFAULT_PLAYER_PROPS: Sequence[str] = (
    "X",
    "Y",
    "Z",
    "health",
    "armor_value",
    "kills_total",
    "deaths_total",
    "assists_total",
    "damage_total",
    "utility_damage_total",
    "headshot_kills_total",
    "score",
    "mvps",
    "active_weapon_name",
    "inventory",
    "balance",
    "current_equip_value",
    "cash_spent_this_round",
    "is_alive",
    "team_name",
    "last_place_name",
    "flash_duration",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "pitch",
    "yaw",
)

DEFAULT_OTHER_PROPS: Sequence[str] = (
    "game_time",
    "is_bomb_planted",
    "which_bomb_zone",
    "is_freeze_period",
    "is_warmup_period",
    "total_rounds_played",
    "is_match_started",
)


def parse_demo_to_zip(
    demo_path: Path,
    output_folder: Path,
    *,
    player_props: Optional[Iterable[str]] = None,
    other_props: Optional[Iterable[str]] = None,
) -> Path:
    """
    Parse a demo file and compress the parsed artefacts into a zip archive.

    Parameters
    ----------
    demo_path:
        Path to the `.dem` file to parse.
    output_folder:
        Directory where the compressed artefact should be written.
    player_props / other_props:
        Optional overrides for the parsing configuration. When omitted, the
        defaults defined above are used.

    Returns
    -------
    Path
        Path to the generated `.zip` archive.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    demo = Demo(path=demo_path, verbose=False)
    demo.parse(
        player_props=list(player_props or DEFAULT_PLAYER_PROPS),
        other_props=list(other_props or DEFAULT_OTHER_PROPS),
    )

    events = getattr(demo, "events", None)
    if isinstance(events, dict) and "player_sound" not in events:
        events["player_sound"] = pl.DataFrame(schema={"tick": pl.Int64})

    demo.compress(outpath=output_folder)
    return output_folder / f"{demo_path.stem}.zip"


__all__ = [
    "DEFAULT_PLAYER_PROPS",
    "DEFAULT_OTHER_PROPS",
    "parse_demo_to_zip",
]
