"""
Win Probability Curve Visualization
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from round_temporal_model import (
    prepare_round_features,
    add_temporal_features,
    REGULATION_HALF_ROUNDS,
    REGULATION_TOTAL_ROUNDS,
    OVERTIME_HALF_ROUNDS,
    OVERTIME_BLOCK_ROUNDS,
)

warnings.filterwarnings('ignore')

ROUNDS_TO_WIN = REGULATION_HALF_ROUNDS + 1

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def team_on_ct(round_num: int) -> int:
    """Return 1 if Team1 is on CT side this round, else 2."""
    if round_num <= REGULATION_HALF_ROUNDS:
        return 1
    if round_num <= REGULATION_TOTAL_ROUNDS:
        return 2

    # Overtime: alternate every overtime half (MR3)
    ot_round_index = round_num - REGULATION_TOTAL_ROUNDS - 1  # zero-based OT rounds
    block = ot_round_index // OVERTIME_HALF_ROUNDS
    return 1 if block % 2 == 0 else 2


def compute_map_odds_ratio(rounds_df: pd.DataFrame, matches_df: pd.DataFrame) -> dict:
    """Compute CT odds ratio per map based on historical outcomes."""
    merged = rounds_df.merge(
        matches_df[['match_id', 'map_name']],
        on='match_id',
        how='left'
    )
    merged['ct_win'] = (merged['round_winner'] == 'ct').astype(int)
    map_ct_rate = merged.groupby('map_name')['ct_win'].mean()
    eps = 1e-6
    odds_ratio = ((map_ct_rate + eps) / (1 - map_ct_rate + eps)).to_dict()
    odds_ratio['unknown'] = 1.0
    return odds_ratio


def train_model_probabilities(features_df: pd.DataFrame, map_odds_ratio: dict) -> pd.DataFrame:
    """Train LightGBM on temporal features and return probabilities per round."""
    df = features_df.reset_index(drop=True).copy()
    exclude_cols = ['match_id', 'round_num', 'round_winner', 'map_name']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df['round_winner'].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        objective='binary',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    ratios = df['map_name'].map(map_odds_ratio).fillna(1.0).values
    odds = probs / np.clip(1 - probs, 1e-6, None)
    adjusted_odds = odds * ratios
    adjusted_probs = adjusted_odds / (1 + adjusted_odds)
    df['ct_win_prob'] = np.clip(adjusted_probs, 1e-6, 1 - 1e-6)
    return df


def build_match_probability_frames(
    df_with_probs: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Create per-match probability DataFrames ready for visualization."""
    team_lookup = matches_df.set_index('match_id')[['team1_name', 'team2_name']]
    results = {}
    all_frames = []

    for match_id, match_rows in df_with_probs.groupby('match_id'):
        match_rows = match_rows.sort_values('round_num').reset_index(drop=True)
        team1_label = team_lookup.get('team1_name', pd.Series(dtype=str)).get(match_id, 'Team 1')
        team2_label = team_lookup.get('team2_name', pd.Series(dtype=str)).get(match_id, 'Team 2')
        team1_label = str(team1_label) if str(team1_label).strip() else 'Team 1'
        team2_label = str(team2_label) if str(team2_label).strip() else 'Team 2'

        team1_probs = []
        team2_probs = []
        team1_scores = []
        team2_scores = []
        team_round_winners = []
        t1_score = 0
        t2_score = 0

        for _, row in match_rows.iterrows():
            round_num = int(row['round_num'])
            ct_prob = row['ct_win_prob']
            ct_team = team_on_ct(round_num)
            team1_prob = ct_prob if ct_team == 1 else 1 - ct_prob
            team2_prob = 1 - team1_prob

            ct_won = row['round_winner'] == 1
            winner_team = ct_team if ct_won else (2 if ct_team == 1 else 1)
            if winner_team == 1:
                t1_score += 1
            else:
                t2_score += 1

            team1_probs.append(team1_prob)
            team2_probs.append(team2_prob)
            team1_scores.append(t1_score)
            team2_scores.append(t2_score)
            team_round_winners.append(winner_team)

        result_df = pd.DataFrame({
            'round': match_rows['round_num'].values,
            'team1_win_prob': team1_probs,
            'team2_win_prob': team2_probs,
            'team1_score': team1_scores,
            'team2_score': team2_scores,
            'team_round_winner': team_round_winners,
            'match_id': match_id,
            'team1_label': team1_label,
            'team2_label': team2_label,
            'map_name': match_rows['map_name'].iloc[0],
        })

        results[match_id] = (result_df, team1_label, team2_label)
        all_frames.append(result_df)

    all_probs_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return results, all_probs_df


def plot_single_match_probability(match_df, match_id, team1_label="Team 1", team2_label="Team 2", save_path=None):
    """
    Plot win probability curve for a single match.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Win probability curve
    rounds = match_df['round']
    team1_prob = match_df['team1_win_prob'] * 100
    team2_prob = match_df['team2_win_prob'] * 100
    
    # Main probability lines
    ax1.plot(rounds, team1_prob, 'b-', linewidth=2.5, label=f'{team1_label} Win %', alpha=0.8)
    ax1.plot(rounds, team2_prob, 'r-', linewidth=2.5, label=f'{team2_label} Win %', alpha=0.8)
    
    # 50% line
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # Shade regions
    ax1.fill_between(rounds, 50, team1_prob, where=(team1_prob >= 50), 
                     alpha=0.2, color='blue', label=f'{team1_label} Favored')
    ax1.fill_between(rounds, 50, team2_prob, where=(team2_prob >= 50), 
                     alpha=0.2, color='red', label=f'{team2_label} Favored')
    
    # Mark important rounds
    for idx, row in match_df.iterrows():
        if row['round'] in [1, REGULATION_HALF_ROUNDS + 1]:  # Pistol rounds
            ax1.axvline(x=row['round'], color='green', linestyle=':', alpha=0.3)
            ax1.text(row['round'], 95, 'Pistol', rotation=90, fontsize=8, alpha=0.5)
        elif row['round'] == REGULATION_HALF_ROUNDS:  # Half switch
            ax1.axvline(x=row['round'], color='orange', linestyle='--', alpha=0.5)
            ax1.text(row['round'], 95, 'Half', rotation=90, fontsize=8, alpha=0.5)
    
    # Labels and formatting
    ax1.set_ylabel('Win Probability (%)', fontsize=12)
    ax1.set_title(f'Match {match_id}: Win Probability Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(rounds) + 0.5)
    ax1.set_ylim(0, 100)
    
    # Score progression bar
    for idx, row in match_df.iterrows():
        winner = row.get('team_round_winner', 1)
        color = 'blue' if winner == 1 else 'red'
        ax2.barh(0, 1, left=row['round']-1, height=0.8, color=color, alpha=0.7)
    
    # Score annotations
    final_team1 = match_df.iloc[-1]['team1_score']
    final_team2 = match_df.iloc[-1]['team2_score']
    ax2.text(len(rounds)/2, 0, f'Final: {team1_label} {final_team1} - {final_team2} {team2_label}', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax2.set_xlim(0.5, len(rounds) + 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_yticks([])
    ax2.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_multiple_matches_comparison(matches_data, title="Match Win Probability Comparisons"):
    """
    Compare win probability curves for multiple matches.
    """
    n_matches = len(matches_data)
    fig, axes = plt.subplots(n_matches, 1, figsize=(14, 4*n_matches), sharex=True)
    
    if n_matches == 1:
        axes = [axes]
    
    for idx, (match_id, (match_df, team1_label, team2_label)) in enumerate(matches_data.items()):
        ax = axes[idx]
        
        rounds = match_df['round']
        team1_prob = match_df['team1_win_prob'] * 100
        
        # Plot probability
        ax.plot(rounds, team1_prob, 'b-', linewidth=2, alpha=0.8)
        ax.fill_between(rounds, 50, team1_prob, where=(team1_prob >= 50), 
                        alpha=0.2, color='blue')
        ax.fill_between(rounds, 50, team1_prob, where=(team1_prob < 50), 
                        alpha=0.2, color='red')
        
        # 50% line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # Format
        final_team1 = match_df.iloc[-1]['team1_score']
        final_team2 = match_df.iloc[-1]['team2_score']
        ax.set_ylabel(f'{team1_label} Win %', fontsize=10)
        ax.set_title(f'Match {match_id} ({team1_label} {final_team1}-{final_team2} {team2_label})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    axes[-1].set_xlabel('Round', fontsize=12)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_comeback_matches(all_probs_df, threshold=0.8):
    """
    Identify matches with significant comebacks.
    """
    comebacks = []
    
    for match_id in all_probs_df['match_id'].unique():
        match_df = all_probs_df[all_probs_df['match_id'] == match_id]
        
        # Find maximum probability difference
        team1_probs = match_df['team1_win_prob'].values
        
        # Check for CT comeback (low to high)
        min_prob_early = team1_probs[:10].min() if len(team1_probs) > 10 else team1_probs.min()
        max_prob_late = team1_probs[-5:].max() if len(team1_probs) > 5 else team1_probs.max()
        
        if min_prob_early < (1 - threshold) and max_prob_late > 0.5:
            comebacks.append({
                'match_id': match_id,
                'type': 'CT Comeback',
                'swing': max_prob_late - min_prob_early,
                'low_point': min_prob_early,
                'final_prob': team1_probs[-1]
            })
        
        # Check for T comeback
        elif max_prob_late < threshold and min_prob_early > 0.5:
            comebacks.append({
                'match_id': match_id,
                'type': 'Team2 Comeback',
                'swing': min_prob_early - max_prob_late,
                'low_point': 1 - max_prob_late,
                'final_prob': 1 - team1_probs[-1]
            })
    
    return pd.DataFrame(comebacks)


def main():
    """
    Main execution to generate win probability visualizations.
    """
    
    print("="*80)
    print("WIN PROBABILITY VISUALIZATION")
    print("="*80)
    
    data_dir = Path("clean_dataset")
    
    print("\nLoading match data...")
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    round_players_df = pd.read_csv(data_dir / "round_players.csv")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    
    print("Preparing temporal features...")
    features_df = prepare_round_features(rounds_df, round_players_df, matches_df)
    features_df = add_temporal_features(features_df)
    
    print("Training LightGBM round model...")
    map_odds_ratio = compute_map_odds_ratio(rounds_df, matches_df)
    features_with_probs = train_model_probabilities(features_df, map_odds_ratio)
    
    print("Building probability curves...")
    match_frames, all_probs_df = build_match_probability_frames(features_with_probs, matches_df)
    
    if all_probs_df.empty:
        print("No rounds available. Exiting.")
        return
    
    # Visualize sample matches
    print("\nGenerating visualizations...")
    first_match_id = next(iter(match_frames))
    sample_df, sample_team1, sample_team2 = match_frames[first_match_id]
    plot_single_match_probability(
        sample_df,
        first_match_id,
        sample_team1,
        sample_team2,
        'win_probability_single.png'
    )
    
    sample_matches = {
        match_id: match_frames[match_id]
        for match_id in list(match_frames.keys())[:3]
    }
    plot_multiple_matches_comparison(sample_matches)
    
    # Analyze comebacks
    print("\nAnalyzing comeback matches...")
    comebacks_df = analyze_comeback_matches(all_probs_df, threshold=0.75)
    
    if not comebacks_df.empty:
        print(f"\nFound {len(comebacks_df)} comeback matches:")
        print(comebacks_df.head())
        comeback_match = comebacks_df.iloc[0]['match_id']
        comeback_df, comeback_team1, comeback_team2 = match_frames[comeback_match]
        plot_single_match_probability(
            comeback_df,
            comeback_match,
            comeback_team1,
            comeback_team2,
            'comeback_example.png'
        )
    
    # Statistical summary
    print("\n" + "="*50)
    print("PROBABILITY STATISTICS")
    print("="*50)
    
    for match_id in all_probs_df['match_id'].unique():
        match_df = all_probs_df[all_probs_df['match_id'] == match_id]
        prob_range = match_df['team1_win_prob'].max() - match_df['team1_win_prob'].min()
        print(f"Match {match_id}: Probability range = {prob_range:.1%}")
    
    # Save results
    all_probs_df.to_csv('win_probabilities.csv', index=False)
    print("\nWin probability analysis complete!")
    print("Files saved:")
    print("  - win_probabilities.csv")
    print("  - win_probability_single.png")
    print("  - comeback_example.png (if comebacks found)")


if __name__ == "__main__":
    main()
