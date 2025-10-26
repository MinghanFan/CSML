"""
Win Probability Curve Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from scipy.stats import beta
import warnings

warnings.filterwarnings('ignore')

REGULATION_HALF_ROUNDS = 12
ROUNDS_TO_WIN = REGULATION_HALF_ROUNDS + 1
REGULATION_TOTAL_ROUNDS = REGULATION_HALF_ROUNDS * 2
OVERTIME_HALF_ROUNDS = 3
OVERTIME_BLOCK_ROUNDS = OVERTIME_HALF_ROUNDS * 2

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


class WinProbabilityCalculator:
    """
    Calculate match win probability based on round outcomes and state.
    Uses both statistical models and domain knowledge.
    """
    
    def __init__(self):
        self.round_importance_weights = self.get_round_importance_weights()
    
    def get_round_importance_weights(self):
        """Define importance weights for different rounds"""
        weights = {}
        max_weight_round = REGULATION_TOTAL_ROUNDS + OVERTIME_BLOCK_ROUNDS
        second_half_pistol = REGULATION_HALF_ROUNDS + 1
        for r in range(1, max_weight_round + 1):
            if r == 1 or r == second_half_pistol:  # Pistol rounds
                weights[r] = 1.5
            elif r in [2, 3, second_half_pistol + 1, second_half_pistol + 2]:  # Anti-eco after pistol
                weights[r] = 1.2
            elif r in [REGULATION_HALF_ROUNDS, REGULATION_TOTAL_ROUNDS]:  # Last round of half
                weights[r] = 1.3
            else:
                weights[r] = 1.0
        return weights
    
    def calculate_base_probability(self, team1_rounds, team2_rounds):
        """
        Calculate base win probability using Beta distribution.
        This provides a Bayesian estimate with uncertainty.
        """
        # Add pseudo-counts for Bayesian prior (prevents 0/1 extremes)
        alpha = team1_rounds + 1
        beta_param = team2_rounds + 1
        
        # Expected win probability
        p_win = alpha / (alpha + beta_param)
        
        rounds_remaining = max(ROUNDS_TO_WIN - max(team1_rounds, team2_rounds), 0)
        if rounds_remaining > 0:
            # More uncertainty with more rounds to play
            uncertainty_factor = rounds_remaining / REGULATION_TOTAL_ROUNDS
            p_win = p_win * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
        
        return p_win
    
    def adjust_for_momentum(self, base_prob, streak, recent_performance):
        """
        Adjust probability based on momentum factors.
        """
        # Streak adjustment (capped to prevent extreme swings)
        streak_adjustment = np.tanh(streak / 5) * 0.1
        
        # Recent performance adjustment
        perf_adjustment = recent_performance * 0.05
        
        # Apply adjustments
        adjusted_prob = base_prob + streak_adjustment + perf_adjustment
        
        # Ensure probability stays in [0.05, 0.95] range
        return np.clip(adjusted_prob, 0.05, 0.95)
    
    def calculate_match_win_probability(self, round_data):
        """
        Calculate evolving win probability for a match.
        """
        probabilities = []
        
        for idx, row in round_data.iterrows():
            # Base probability from score
            base_prob = self.calculate_base_probability(
                row['team1_score'], row['team2_score']
            )
            
            # Momentum adjustments
            if 'team1_streak' in row:
                adjusted_prob = self.adjust_for_momentum(
                    base_prob,
                    row.get('team1_streak', 0),
                    row.get('team_recent_perf', 0)
                )
            else:
                adjusted_prob = base_prob
            
            probabilities.append({
                'round': row['round_num'],
                'team1_win_prob': adjusted_prob,
                'team2_win_prob': 1 - adjusted_prob,
                'team1_score': row['team1_score'],
                'team2_score': row['team2_score'],
                'team_round_winner': row.get('team_round_winner', 'unknown')
            })
        
        return pd.DataFrame(probabilities)


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
    
    # Load data
    data_dir = Path("clean_dataset")
    
    print("\nLoading match data...")
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    
    # Initialize calculator
    calc = WinProbabilityCalculator()
    
    # Calculate probabilities for all matches
    all_probabilities = []
    
    print("\nCalculating win probabilities...")
    for match_id in rounds_df['match_id'].unique()[:10]:  # Process first 10 matches
        match_rounds = (
            rounds_df[rounds_df['match_id'] == match_id]
            .sort_values('round_num')
            .reset_index(drop=True)
        )
        
        match_meta = matches_df[matches_df['match_id'] == match_id].iloc[0]
        team1_label = match_meta.get('team1_name', 'Team 1')
        team2_label = match_meta.get('team2_name', 'Team 2')

        team1_scores = []
        team2_scores = []
        team1_wins = []
        team1_streaks = []
        team_round_winners = []
        streak = 0
        team1_score = 0
        team2_score = 0

        for _, row in match_rounds.iterrows():
            ct_team = team_on_ct(int(row['round_num']))
            team1_on_ct = ct_team == 1
            winner_side = row['round_winner']
            team1_won = (winner_side == 'ct' and team1_on_ct) or (winner_side == 't' and not team1_on_ct)

            if team1_won:
                team1_score += 1
                streak = max(0, streak) + 1
            else:
                team2_score += 1
                streak = min(0, streak) - 1

            team1_scores.append(team1_score)
            team2_scores.append(team2_score)
            team1_wins.append(1 if team1_won else 0)
            team1_streaks.append(streak)
            team_round_winners.append(1 if team1_won else 2)

        match_rounds['team1_score'] = team1_scores
        match_rounds['team2_score'] = team2_scores
        match_rounds['team1_streak'] = team1_streaks
        match_rounds['team1_win'] = team1_wins
        match_rounds['team_round_winner'] = team_round_winners
        match_rounds['team_recent_perf'] = (
            match_rounds['team1_win'].rolling(5, min_periods=1).mean() - 0.5
        ) * 2

        prob_df = calc.calculate_match_win_probability(match_rounds)
        prob_df['match_id'] = match_id
        prob_df['team1_label'] = team1_label
        prob_df['team2_label'] = team2_label
        all_probabilities.append(prob_df)
    
    all_probs_df = pd.concat(all_probabilities, ignore_index=True)
    
    # Visualize sample matches
    print("\nGenerating visualizations...")
    
    # Single match detailed view
    sample_match = all_probs_df['match_id'].iloc[0]
    sample_df = all_probs_df[all_probs_df['match_id'] == sample_match]
    sample_team1 = sample_df['team1_label'].iloc[0]
    sample_team2 = sample_df['team2_label'].iloc[0]
    plot_single_match_probability(
        sample_df,
        sample_match,
        sample_team1,
        sample_team2,
        'win_probability_single.png'
    )
    
    # Multiple matches comparison
    sample_matches = {}
    for match_id in all_probs_df['match_id'].unique()[:3]:
        df_match = all_probs_df[all_probs_df['match_id'] == match_id]
        sample_matches[match_id] = (
            df_match,
            df_match['team1_label'].iloc[0],
            df_match['team2_label'].iloc[0],
        )
    
    plot_multiple_matches_comparison(sample_matches)
    
    # Analyze comebacks
    print("\nAnalyzing comeback matches...")
    comebacks_df = analyze_comeback_matches(all_probs_df, threshold=0.75)
    
    if len(comebacks_df) > 0:
        print(f"\nFound {len(comebacks_df)} comeback matches:")
        print(comebacks_df.head())
        
        # Plot a comeback match if found
        if len(comebacks_df) > 0:
            comeback_match = comebacks_df.iloc[0]['match_id']
            comeback_df = all_probs_df[all_probs_df['match_id'] == comeback_match]
            team1_label = comeback_df['team1_label'].iloc[0]
            team2_label = comeback_df['team2_label'].iloc[0]
            plot_single_match_probability(
                comeback_df,
                comeback_match,
                team1_label,
                team2_label,
                'comeback_example.png'
            )
    
    # Statistical summary
    print("\n" + "="*50)
    print("PROBABILITY STATISTICS")
    print("="*50)
    
    # Average probability swing per match
    for match_id in all_probs_df['match_id'].unique():
        match_df = all_probs_df[all_probs_df['match_id'] == match_id]
        prob_range = match_df['team1_win_prob'].max() - match_df['team1_win_prob'].min()
        print(f"Match {match_id}: Probability range = {prob_range:.1%}")
    
    # Save results
    all_probs_df.to_csv('win_probabilities.csv', index=False)
    print("\n Win probability analysis complete!")
    print("Files saved:")
    print("  - win_probabilities.csv")
    print("  - win_probability_single.png")
    print("  - comeback_example.png (if comebacks found)")


if __name__ == "__main__":
    main()
