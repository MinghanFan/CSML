from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

CLEAN = Path("clean_dataset")

def read_parquet_safe(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()

def main():
    rounds_csv = CLEAN / "rounds.csv"
    if not rounds_csv.exists():
        print("[error] clean_dataset/rounds.csv not found. Your existing rounds parser must produce it so we can build labels (y).", file=sys.stderr)
        sys.exit(1)

    # Load rounds with winners to produce labels
    rounds = pd.read_csv(rounds_csv)
    # Expect columns: match_id, round_num, round_winner (T/CT), optional map_name, scores...
    if "round_winner" not in rounds.columns:
        print("[error] rounds.csv does not contain 'round_winner'. Cannot create labels (y).", file=sys.stderr)
        sys.exit(1)

    # Normalize winner (upper)
    rounds["round_winner"] = rounds["round_winner"].astype(str).str.upper()

    # Load event tables
    kills_pl     = read_parquet_safe(CLEAN / "kills.parquet")
    damages_pl   = read_parquet_safe(CLEAN / "damages.parquet")
    grenades_pl  = read_parquet_safe(CLEAN / "grenades.parquet")
    bomb_pl      = read_parquet_safe(CLEAN / "bomb.parquet")
    shots_pl     = read_parquet_safe(CLEAN / "shots.parquet")
    smokes_pl    = read_parquet_safe(CLEAN / "smokes.parquet")
    infernos_pl  = read_parquet_safe(CLEAN / "infernos.parquet")

    # Convert to pandas for flexible groupby operations
    kills    = kills_pl.to_pandas()    if kills_pl.height    else pd.DataFrame(columns=["match_id","round_num"])
    damages  = damages_pl.to_pandas()  if damages_pl.height  else pd.DataFrame(columns=["match_id","round_num"])
    grenades = grenades_pl.to_pandas() if grenades_pl.height else pd.DataFrame(columns=["match_id","round_num"])
    bomb     = bomb_pl.to_pandas()     if bomb_pl.height     else pd.DataFrame(columns=["match_id","round_num"])
    shots    = shots_pl.to_pandas()    if shots_pl.height    else pd.DataFrame(columns=["match_id","round_num"])
    smokes   = smokes_pl.to_pandas()   if smokes_pl.height   else pd.DataFrame(columns=["match_id","round_num"])
    infernos = infernos_pl.to_pandas() if infernos_pl.height else pd.DataFrame(columns=["match_id","round_num"])

    # Standardize side columns that might appear
    def up(x):
        return x if pd.isna(x) else str(x).upper()
    for df, cols in [
        (kills, ["attacker_side","victim_side"]),
        (damages, ["attacker_side","victim_side"]),
        (grenades, ["thrower_side"]),
        (shots, ["shooter_side"]),
        (smokes, ["thrower_side"]),
        (infernos, ["thrower_side"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].map(up)

    # -------------
    # KILLS-derived
    # -------------
    # kills per side
    kills_side = (kills.groupby(["match_id","round_num","attacker_side"], dropna=False)
                        .size().rename("kills").reset_index().rename(columns={"attacker_side":"side"}))

    # headshots per side
    head_mask = kills.get("hitgroup", pd.Series(dtype=object)).isin(["head","HEAD","headshot","HEADSHOT"])
    hs_side = (kills[head_mask].groupby(["match_id","round_num","attacker_side"], dropna=False)
                       .size().rename("headshots").reset_index().rename(columns={"attacker_side":"side"}))

    # deaths per side (victims)
    deaths_side = (kills.groupby(["match_id","round_num","victim_side"], dropna=False)
                         .size().rename("deaths").reset_index().rename(columns={"victim_side":"side"}))

    # first kill flag per side (which side got the earliest kill in the round)
    fk = None
    if not kills.empty and "tick" in kills.columns:
        first = (kills.sort_values(["match_id","round_num","tick"])
                      .groupby(["match_id","round_num"], as_index=False).first())
        first["side"] = first["attacker_side"].map(up)
        first["first_kill"] = 1
        fk = first[["match_id","round_num","side","first_kill"]]

    # ---------------
    # DAMAGES-derived
    # ---------------
    dmg_side = (damages.groupby(["match_id","round_num","attacker_side"], dropna=False)["dmg_health"]
                      .sum().rename("total_damage").reset_index().rename(columns={"attacker_side":"side"}))

    # Utility damage (HE, molotov, incendiary)
    util_weapons = {"HEGRENADE","HE","HE_GRENADE","MOLOTOV","INCGRENADE","INCENDIARY","INC"}
    if "weapon" in damages.columns:
        w = damages["weapon"].astype(str).str.upper()
        damages["__is_util"] = w.isin(util_weapons)
        dmg_util_side = (damages[damages["__is_util"]].groupby(["match_id","round_num","attacker_side"], dropna=False)["dmg_health"]
                                .sum().rename("util_damage").reset_index().rename(columns={"attacker_side":"side"}))
    else:
        dmg_util_side = pd.DataFrame(columns=["match_id","round_num","side","util_damage"])

    # ---------------
    # GRENADES counts
    # ---------------
    def count_nade(name_set: set[str], out_col: str):
        if grenades.empty or "nade_type" not in grenades.columns:
            return pd.DataFrame(columns=["match_id","round_num","side",out_col])
        t = grenades.copy()
        t["nade_type"] = t["nade_type"].astype(str).str.upper()
        t = t[t["nade_type"].isin(name_set)]
        if t.empty:
            return pd.DataFrame(columns=["match_id","round_num","side",out_col])
        grp = (t.groupby(["match_id","round_num","thrower_side"], dropna=False)
                 .size().rename(out_col).reset_index().rename(columns={"thrower_side":"side"}))
        return grp

    he_cnt   = count_nade({"HE","HEGRENADE","HE_GRENADE"}, "he_count")
    flash_cnt= count_nade({"FLASH","FLASHBANG"}, "flash_count")
    smoke_cnt= count_nade({"SMOKE","SMOKEGRENADE","SMOKE_GRENADE"}, "smoke_count")
    mol_cnt  = count_nade({"MOLOTOV","INCGRENADE","INCENDIARY"}, "molotov_inc_count")

    # ---------------
    # SHOTS / SMOKES / INFERNOS counts
    # ---------------
    if "shooter_side" in shots.columns:
        shots_cnt = (shots.groupby(["match_id","round_num","shooter_side"], dropna=False)
                          .size().rename("shots_count").reset_index().rename(columns={"shooter_side":"side"}))
    else:
        shots_cnt = pd.DataFrame(columns=["match_id","round_num","side","shots_count"])

    if "thrower_side" in smokes.columns:
        smokes_cnt = (smokes.groupby(["match_id","round_num","thrower_side"], dropna=False)
                           .size().rename("smokes_count").reset_index().rename(columns={"thrower_side":"side"}))
    else:
        smokes_cnt = pd.DataFrame(columns=["match_id","round_num","side","smokes_count"])

    if "thrower_side" in infernos.columns:
        infernos_cnt = (infernos.groupby(["match_id","round_num","thrower_side"], dropna=False)
                             .size().rename("infernos_count").reset_index().rename(columns={"thrower_side":"side"}))
    else:
        infernos_cnt = pd.DataFrame(columns=["match_id","round_num","side","infernos_count"])

    # ---------------
    # BOMB events → round-level flags
    # ---------------
    bomb_flags = None
    if not bomb.empty and {"event","match_id","round_num"}.issubset(bomb.columns):
        b = bomb.copy()
        b["event"] = b["event"].astype(str).str.lower()
        round_flag = (b.pivot_table(index=["match_id","round_num"],
                                    columns="event",
                                    values="tick",
                                    aggfunc="min"))
        round_flag = round_flag.rename(columns=lambda c: f"bomb_{c}")
        round_flag = round_flag.reset_index()
        # Map to per-side flags
        # T plants/explodes; CT defuses
        round_flag["t_planted"]  = (round_flag.get("bomb_plant").notna()).astype(int) if "bomb_plant" in round_flag else 0
        round_flag["ct_defused"] = (round_flag.get("bomb_defuse").notna()).astype(int) if "bomb_defuse" in round_flag else 0
        round_flag["t_exploded"] = (round_flag.get("bomb_explode").notna()).astype(int) if "bomb_explode" in round_flag else 0
        bomb_flags = round_flag[["match_id","round_num","t_planted","ct_defused","t_exploded"]]
    else:
        bomb_flags = pd.DataFrame(columns=["match_id","round_num","t_planted","ct_defused","t_exploded"])

    # ----------------------------------
    # Merge all side-level feature tables
    # ----------------------------------
    base = pd.DataFrame(columns=["match_id","round_num","side"]).astype({"match_id":str,"round_num":int,"side":str})
    pieces = [kills_side, hs_side, deaths_side, fk, dmg_side, dmg_util_side,
              he_cnt, flash_cnt, smoke_cnt, mol_cnt, shots_cnt, smokes_cnt, infernos_cnt]
    for p in pieces:
        if p is None or p.empty:
            continue
        base = base.merge(p, on=["match_id","round_num","side"], how="outer")

    # Fill NA with 0 for counts
    for c in ["kills","headshots","deaths","first_kill","total_damage","util_damage",
              "he_count","flash_count","smoke_count","molotov_inc_count",
              "shots_count","smokes_count","infernos_count"]:
        if c in base.columns:
            base[c] = base[c].fillna(0)

    # Survivors ≈ 5 - deaths (cap to [0,5])
    if "deaths" in base.columns:
        base["survivors"] = (5 - base["deaths"].astype(float)).clip(lower=0, upper=5).astype(int)
    else:
        base["survivors"] = np.nan

    # Attach bomb flags by round, then map to side-specific columns
    if not bomb_flags.empty:
        base = base.merge(bomb_flags, on=["match_id","round_num"], how="left")
        # Per-side mapping
        base["bomb_planted_side"] = np.where(base["side"]=="T", base["t_planted"], 0)
        base["bomb_defused_side"] = np.where(base["side"]=="CT", base["ct_defused"], 0)
        base["bomb_exploded_side"]= np.where(base["side"]=="T", base["t_exploded"], 0)
        base = base.drop(columns=["t_planted","ct_defused","t_exploded"], errors="ignore")
    else:
        base["bomb_planted_side"] = 0
        base["bomb_defused_side"] = 0
        base["bomb_exploded_side"]= 0

    # ----------------------------------
    # Add labels y via rounds.csv winner
    # ----------------------------------
    lab = rounds[["match_id","round_num","round_winner"]].copy()
    # Expand to per-side rows
    # y=1 if side == round_winner; y=0 otherwise
    base = base.merge(lab, on=["match_id","round_num"], how="left")
    base["y"] = (base["side"].astype(str).str.upper() == base["round_winner"].astype(str).str.upper()).astype(int)
    base = base.drop(columns=["round_winner"])

    # Sort & save
    base = base.sort_values(["match_id","round_num","side"]).reset_index(drop=True)

    out_csv = CLEAN / "features_round_team.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_csv, index=False)
    print(f"[ok] Wrote {out_csv} with {len(base)} rows and {base.shape[1]} columns.")

if __name__ == "__main__":
    main()
