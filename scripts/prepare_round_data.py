"""
Data Preparation for Round Temporal Model
"""

import pandas as pd
import numpy as np
from pathlib import Path

REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = REGULATION_HALF_ROUNDS * 2
OVERTIME_HALF_ROUNDS = 3
OVERTIME_BLOCK_ROUNDS = OVERTIME_HALF_ROUNDS * 2


def prepare_minimal_round_data():
    """
    Prepare round data from existing CSV files when detailed event data is not available.
    Uses rounds.csv and round_players.csv to create features.
    """
    
    print("="*80)
    print("PREPARING DATA FOR TEMPORAL MODEL")
    print("="*80)
    
    data_dir = Path("clean_dataset")
    
    # Load available data
    print("\nLoading existing CSV files...")
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    round_players_df = pd.read_csv(data_dir / "round_players.csv")
    match_players_df = pd.read_csv(data_dir / "match_players.csv")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    
    print(f"Loaded {len(rounds_df)} rounds from {rounds_df['match_id'].nunique()} matches")
    
    # Create simplified features from available data
    print("\nCreating features from available data...")
    
    features_list = []
    
    for match_id in rounds_df['match_id'].unique():
        match_rounds = rounds_df[rounds_df['match_id'] == match_id].sort_values('round_num')
        match_round_players = round_players_df[round_players_df['match_id'] == match_id]
        
        for idx, round_row in match_rounds.iterrows():
            round_num = round_row['round_num']
            round_players = match_round_players[match_round_players['round_num'] == round_num]
            
            # Aggregate by team
            ct_players = round_players[round_players['team'] == 'ct']
            t_players = round_players[round_players['team'] == 't']
            
            # Create features
            features = {
                'match_id': match_id,
                'round_num': round_num,
                'round_winner': 1 if round_row['round_winner'] == 'ct' else 0,
                
                # From rounds table
                'ct_equipment_value': round_row['ct_equipment_value'],
                't_equipment_value': round_row['t_equipment_value'],
                'equipment_diff': round_row['ct_equipment_value'] - round_row['t_equipment_value'],
                'ct_alive_end': round_row['ct_players_alive_end'],
                't_alive_end': round_row['t_players_alive_end'],
                'alive_diff_end': round_row['ct_players_alive_end'] - round_row['t_players_alive_end'],
                'bomb_planted': 1 if round_row['bomb_planted'] else 0,
                'round_duration': round_row['round_duration'],
                
                # From round_players aggregated
                'ct_total_kills': ct_players['kills'].sum() if len(ct_players) > 0 else 0,
                't_total_kills': t_players['kills'].sum() if len(t_players) > 0 else 0,
                'ct_total_deaths': ct_players['deaths'].sum() if len(ct_players) > 0 else 0,
                't_total_deaths': t_players['deaths'].sum() if len(t_players) > 0 else 0,
                'ct_total_damage': ct_players['damage'].sum() if len(ct_players) > 0 else 0,
                't_total_damage': t_players['damage'].sum() if len(t_players) > 0 else 0,
                'ct_avg_damage': ct_players['damage'].mean() if len(ct_players) > 0 else 0,
                't_avg_damage': t_players['damage'].mean() if len(t_players) > 0 else 0,
                'ct_survivors': ct_players['survived'].sum() if len(ct_players) > 0 else 0,
                't_survivors': t_players['survived'].sum() if len(t_players) > 0 else 0,
                'ct_headshots': ct_players['headshots'].sum() if len(ct_players) > 0 else 0,
                't_headshots': t_players['headshots'].sum() if len(t_players) > 0 else 0,
                
                # Differences
                'kill_diff': (ct_players['kills'].sum() - t_players['kills'].sum()) if len(ct_players) > 0 and len(t_players) > 0 else 0,
                'damage_diff': (ct_players['damage'].sum() - t_players['damage'].sum()) if len(ct_players) > 0 and len(t_players) > 0 else 0,
                'survivor_diff': (ct_players['survived'].sum() - t_players['survived'].sum()) if len(ct_players) > 0 and len(t_players) > 0 else 0,
                
                # Economy
                'ct_total_equip_value': ct_players['equipment_value'].sum() if len(ct_players) > 0 else 0,
                't_total_equip_value': t_players['equipment_value'].sum() if len(t_players) > 0 else 0,
                'ct_avg_equip_value': ct_players['equipment_value'].mean() if len(ct_players) > 0 else 0,
                't_avg_equip_value': t_players['equipment_value'].mean() if len(t_players) > 0 else 0,
                'ct_total_cash_spent': ct_players['cash_spent'].sum() if len(ct_players) > 0 else 0,
                't_total_cash_spent': t_players['cash_spent'].sum() if len(t_players) > 0 else 0,
                
                # Round context
                'round_index': round_num,
                'is_first_half': 1 if round_num <= REGULATION_HALF_ROUNDS else 0,
                'is_second_half': 1 if (round_num > REGULATION_HALF_ROUNDS and round_num <= REGULATION_TOTAL_ROUNDS) else 0,
                'is_overtime': 1 if round_num > REGULATION_TOTAL_ROUNDS else 0,
                'round_in_half': (
                    round_num
                    if round_num <= REGULATION_HALF_ROUNDS
                    else (
                        round_num - REGULATION_HALF_ROUNDS
                        if round_num <= REGULATION_TOTAL_ROUNDS
                        else ((round_num - REGULATION_TOTAL_ROUNDS - 1) % OVERTIME_HALF_ROUNDS) + 1
                    )
                )
            }
            
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    print(f"Created {len(df)} round records with {len(df.columns)} base features")
    
    # Add momentum features
    print("\nAdding momentum and temporal features...")
    df = df.sort_values(['match_id', 'round_num'])
    
    # Win/loss streaks
    df['ct_won'] = df['round_winner'].astype(int)
    df['t_won'] = 1 - df['ct_won']
    
    # Calculate streaks and scores
    for match_id in df['match_id'].unique():
        match_mask = df['match_id'] == match_id
        match_df = df[match_mask].copy()
        
        # Streaks
        ct_streak = 0
        t_streak = 0
        ct_streaks = []
        t_streaks = []
        
        for ct_win, t_win in zip(match_df['ct_won'], match_df['t_won']):
            if ct_win:
                ct_streak += 1
                t_streak = 0
            else:
                t_streak += 1
                ct_streak = 0
            ct_streaks.append(ct_streak)
            t_streaks.append(t_streak)
        
        df.loc[match_mask, 'ct_streak'] = ct_streaks
        df.loc[match_mask, 't_streak'] = t_streaks
    
    # Shift streaks so they reflect the position entering the round
    df['ct_streak'] = df.groupby('match_id')['ct_streak'].shift(1).fillna(0)
    df['t_streak'] = df.groupby('match_id')['t_streak'].shift(1).fillna(0)
    
    # Cumulative scores
    df['ct_score'] = df.groupby('match_id')['ct_won'].cumsum().shift(1).fillna(0).astype(int)
    df['t_score'] = df.groupby('match_id')['t_won'].cumsum().shift(1).fillna(0).astype(int)
    df['score_diff'] = df['ct_score'] - df['t_score']
    
    # Add lag features for key metrics
    lag_features = ['kill_diff', 'damage_diff', 'equipment_diff', 'survivor_diff', 
                   'ct_streak', 't_streak', 'score_diff']
    lag_rounds = [1, 2, 3, 5]
    
    for col in lag_features:
        for lag in lag_rounds:
            lag_col = f'{col}_lag{lag}'
            df[lag_col] = df.groupby('match_id')[col].shift(lag)
    
    # Rolling averages
    window_sizes = [3, 5]
    for col in ['kill_diff', 'damage_diff', 'equipment_diff']:
        for window in window_sizes:
            roll_col = f'{col}_roll{window}'
            df[roll_col] = df.groupby('match_id')[col].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            # Shift by 1 to avoid leakage
            df[roll_col] = df.groupby('match_id')[roll_col].shift(1)
    
    # Cumulative statistics (shifted to avoid leakage)
    cumsum_cols = ['ct_total_kills', 't_total_kills', 'ct_total_damage', 't_total_damage']
    for col in cumsum_cols:
        cum_col = f'{col}_cumsum'
        df[cum_col] = df.groupby('match_id')[col].cumsum()
        df[cum_col] = df.groupby('match_id')[cum_col].shift(1)
    
    # Add match context from match_players
    print("\nAdding match-level player statistics...")
    
    # Get average player stats per team per match
    team_stats = match_players_df.groupby(['match_id', 'team']).agg({
        'kills': 'mean',
        'deaths': 'mean',
        'adr': 'mean',
        'kd_ratio': 'mean',
        'hsp': 'mean',
        'utility_damage': 'mean',
        'flash_assists': 'mean',
        'first_kills': 'sum',
        'clutches_won': 'sum',
        'performance_score': 'mean'
    }).reset_index()
    
    # Reshape to get CT and T stats
    ct_stats = team_stats[team_stats['team'].str.contains('Team')].copy()
    
    # Since we don't know which team was CT/T, we'll use match outcome
    # This is a simplification - in production you'd track this properly
    for match_id in df['match_id'].unique():
        match_info = matches_df[matches_df['match_id'] == match_id].iloc[0]
        
        # Add some match context (simplified)
        df.loc[df['match_id'] == match_id, 'map'] = match_info['map_name']
        df.loc[df['match_id'] == match_id, 'total_rounds_in_match'] = match_info['total_rounds']
    
    # Drop columns that leak round-t information
    leaky_columns = [
        'ct_alive_end', 't_alive_end', 'alive_diff_end',
        'ct_total_kills', 't_total_kills', 'ct_total_deaths', 't_total_deaths',
        'ct_total_damage', 't_total_damage', 'ct_avg_damage', 't_avg_damage',
        'ct_survivors', 't_survivors', 'ct_headshots', 't_headshots',
        'kill_diff', 'damage_diff', 'survivor_diff',
        'ct_total_equip_value', 't_total_equip_value',
        'ct_avg_equip_value', 't_avg_equip_value',
        'ct_total_cash_spent', 't_total_cash_spent',
        'ct_equipment_value', 't_equipment_value', 'equipment_diff',
        'bomb_planted', 'round_duration',
        'ct_won', 't_won'
    ]
    existing_leaky = [col for col in leaky_columns if col in df.columns]
    if existing_leaky:
        df = df.drop(columns=existing_leaky)
    
    print(f"\nFinal dataset: {len(df)} rounds with {len(df.columns)} features")
    
    # Save prepared data
    output_path = Path("clean_dataset/temporal_rounds_prepared.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved prepared data to {output_path}")
    
    # Show feature summary
    print("\n" + "="*50)
    print("FEATURE SUMMARY")
    print("="*50)
    
    base_features = [col for col in df.columns if 'lag' not in col and 'roll' not in col and 'cumsum' not in col]
    temporal_features = [col for col in df.columns if 'lag' in col or 'roll' in col or 'cumsum' in col]
    
    print(f"Base features: {len(base_features)}")
    print(f"Temporal features: {len(temporal_features)}")
    print(f"Total features: {len(df.columns)}")
    
    print("\nSample of temporal features created:")
    for feat in temporal_features[:10]:
        print(f"  - {feat}")
    
    print("\nTarget distribution:")
    print(df['round_winner'].value_counts())
    print(f"CT win rate: {df['round_winner'].mean():.2%}")
    
    # Check for data quality
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    null_counts = df.isnull().sum()
    high_null_cols = null_counts[null_counts > len(df) * 0.3]
    
    if len(high_null_cols) > 0:
        print("Columns with >30% missing values (expected for lag features in early rounds):")
        for col, count in high_null_cols.items():
            print(f"  - {col}: {count/len(df):.1%} missing")
    else:
        print("No columns with excessive missing values")
    
    print("\n Data preparation for round temporal model complete")
    
    return df


if __name__ == "__main__":
    df = prepare_minimal_round_data()
