"""
Data schema definitions for CS2 dataset.
Defines the structure of all output tables.
"""

import polars as pl
from dataclasses import dataclass

@dataclass
class CS2DataSchema:
    """
    Complete data schema for CS2 analysis.
    All tables and their column definitions.
    """
    
    # === TABLE 1: MATCHES ===
    matches_schema = {
        'match_id': pl.Utf8,
        'demo_file': pl.Utf8,
        'map_name': pl.Utf8,
        'date': pl.Utf8,
        'total_rounds': pl.Int32,
        'team1_name': pl.Utf8,
        'team2_name': pl.Utf8,
        'team1_score': pl.Int32,
        'team2_score': pl.Int32,
        'winner': pl.Utf8,
        'match_duration_seconds': pl.Float32,
    }
    
    # === TABLE 2: PLAYERS ===
    players_schema = {
        'player_id': pl.Utf8,
        'player_name': pl.Utf8,
        'total_matches': pl.Int32,
        'primary_role': pl.Utf8,
        'avg_rank': pl.Float32,
    }
    
    # === TABLE 3: MATCH_PLAYERS (MAIN ML DATASET) ===
    match_players_schema = {
        'match_id': pl.Utf8,
        'player_id': pl.Utf8,
        'player_name': pl.Utf8,
        'team': pl.Utf8,  # 'ct' or 't' (side, not actual team name)
        'role': pl.Utf8,
        
        # Core stats
        'kills': pl.Int32,
        'deaths': pl.Int32,
        'assists': pl.Int32,
        'headshot_kills': pl.Int32,
        'damage': pl.Int32,
        'utility_damage': pl.Int32,
        'enemies_flashed': pl.Int32,
        'flash_assists': pl.Int32,
        
        # Metrics
        'kd_ratio': pl.Float32,
        'kda_ratio': pl.Float32,
        'adr': pl.Float32,
        'hsp': pl.Float32,
        
        # Survival & Impact
        'rounds_survived': pl.Int32,
        'survival_rate': pl.Float32,
        'first_kills': pl.Int32,
        'first_deaths': pl.Int32,
        'clutches_attempted': pl.Int32,
        'clutches_won': pl.Int32,
        'multi_kill_rounds': pl.Int32,
        
        # Utility usage
        'smokes_thrown': pl.Int32,
        'flashes_thrown': pl.Int32,
        'he_thrown': pl.Int32,
        'molotovs_thrown': pl.Int32,
        
        # Economy - FIXED: Changed to averages
        'avg_cash_spent_per_round': pl.Float32,
        'avg_equipment_value_per_round': pl.Float32,
        
        # Performance score (calculated later)
        'performance_score': pl.Float32,
        
        # Outcome
        'won_match': pl.Boolean,
    }
    
    # === TABLE 4: ROUNDS ===
    rounds_schema = {
        'match_id': pl.Utf8,
        'round_num': pl.Int32,
        'round_winner': pl.Utf8,
        'round_end_reason': pl.Utf8,
        'round_duration': pl.Float32,
        'bomb_planted': pl.Boolean,
        'bomb_site': pl.Utf8,
        'ct_equipment_value': pl.Int32,
        't_equipment_value': pl.Int32,
        'ct_players_alive_end': pl.Int32,
        't_players_alive_end': pl.Int32,
    }
    
    # === TABLE 5: ROUND_PLAYERS ===
    round_players_schema = {
        'match_id': pl.Utf8,
        'round_num': pl.Int32,
        'player_id': pl.Utf8,
        'player_name': pl.Utf8,
        'team': pl.Utf8,
        
        # Round stats
        'kills': pl.Int32,
        'deaths': pl.Int32,
        'assists': pl.Int32,
        'damage': pl.Int32,
        'headshots': pl.Int32,
        'survived': pl.Boolean,
        
        # Economy
        'equipment_value': pl.Int32,
        'cash_spent': pl.Int32,
        'money_start': pl.Int32,
        'money_end': pl.Int32,
        
        # Outcome
        'won_round': pl.Boolean,
    }

print("✓ Data schema loaded")