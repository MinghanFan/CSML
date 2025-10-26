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
    
    def calculate_base_probability(self, ct_rounds, t_rounds, rounds_played):
        """
        Calculate base win probability using Beta distribution.
        This provides a Bayesian estimate with uncertainty.
        """
        # Add pseudo-counts for Bayesian prior (prevents 0/1 extremes)
        alpha = ct_rounds + 1
        beta_param = t_rounds + 1
        
        # Expected win probability
        p_win = alpha / (alpha + beta_param)
        
        # Adjust for rounds remaining
        rounds_remaining = max(ROUNDS_TO_WIN - max(ct_rounds, t_rounds), 0)
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
                row['ct_score'], row['t_score'], row['round_num']
            )
            
            # Momentum adjustments
            if 'ct_streak' in row:
                adjusted_prob = self.adjust_for_momentum(
                    base_prob,
                    row.get('ct_streak', 0),
                    row.get('recent_perf', 0)
                )
            else:
                adjusted_prob = base_prob
            
            probabilities.append({
                'round': row['round_num'],
                'ct_win_prob': adjusted_prob,
                't_win_prob': 1 - adjusted_prob,
                'ct_score': row['ct_score'],
                't_score': row['t_score'],
                'round_winner': row.get('round_winner', 'unknown')
            })
        
        return pd.DataFrame(probabilities)


def plot_single_match_probability(match_df, match_id, save_path=None):
    """
    Plot win probability curve for a single match.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Win probability curve
    rounds = match_df['round']
    ct_prob = match_df['ct_win_prob'] * 100
    t_prob = match_df['t_win_prob'] * 100
    
    # Main probability lines
    ax1.plot(rounds, ct_prob, 'b-', linewidth=2.5, label='CT Win Probability', alpha=0.8)
    ax1.plot(rounds, t_prob, 'r-', linewidth=2.5, label='T Win Probability', alpha=0.8)
    
    # 50% line
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # Shade regions
    ax1.fill_between(rounds, 50, ct_prob, where=(ct_prob >= 50), 
                     alpha=0.2, color='blue', label='CT Favored')
    ax1.fill_between(rounds, 50, t_prob, where=(t_prob >= 50), 
                     alpha=0.2, color='red', label='T Favored')
    
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
        winner = row['round_winner']
        ct_win = winner == 1 or str(winner).lower() == 'ct'
        color = 'blue' if ct_win else 'red'
        ax2.barh(0, 1, left=row['round']-1, height=0.8, color=color, alpha=0.7)
    
    # Score annotations
    final_ct = match_df.iloc[-1]['ct_score']
    final_t = match_df.iloc[-1]['t_score']
    ax2.text(len(rounds)/2, 0, f'Final: CT {final_ct} - {final_t} T', 
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
    
    for idx, (match_id, match_df) in enumerate(matches_data.items()):
        ax = axes[idx]
        
        rounds = match_df['round']
        ct_prob = match_df['ct_win_prob'] * 100
        
        # Plot probability
        ax.plot(rounds, ct_prob, 'b-', linewidth=2, alpha=0.8)
        ax.fill_between(rounds, 50, ct_prob, where=(ct_prob >= 50), 
                        alpha=0.2, color='blue')
        ax.fill_between(rounds, 50, ct_prob, where=(ct_prob < 50), 
                        alpha=0.2, color='red')
        
        # 50% line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        # Format
        final_ct = match_df.iloc[-1]['ct_score']
        final_t = match_df.iloc[-1]['t_score']
        ax.set_ylabel('CT Win %', fontsize=10)
        ax.set_title(f'Match {match_id} (CT {final_ct}-{final_t} T)', fontsize=11)
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
        ct_probs = match_df['ct_win_prob'].values
        
        # Check for CT comeback (low to high)
        min_prob_early = ct_probs[:10].min() if len(ct_probs) > 10 else ct_probs.min()
        max_prob_late = ct_probs[-5:].max() if len(ct_probs) > 5 else ct_probs.max()
        
        if min_prob_early < (1 - threshold) and max_prob_late > 0.5:
            comebacks.append({
                'match_id': match_id,
                'type': 'CT Comeback',
                'swing': max_prob_late - min_prob_early,
                'low_point': min_prob_early,
                'final_prob': ct_probs[-1]
            })
        
        # Check for T comeback
        elif max_prob_late < threshold and min_prob_early > 0.5:
            comebacks.append({
                'match_id': match_id,
                'type': 'T Comeback',
                'swing': min_prob_early - max_prob_late,
                'low_point': 1 - max_prob_late,
                'final_prob': 1 - ct_probs[-1]
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
        
        # Calculate cumulative scores
        match_rounds['ct_won'] = (match_rounds['round_winner'] == 'ct').astype(int)
        match_rounds['ct_score'] = match_rounds['ct_won'].cumsum()
        match_rounds['t_score'] = match_rounds.index + 1 - match_rounds['ct_score']
        
        # Calculate streak
        streak = 0
        streaks = []
        for won in match_rounds['ct_won']:
            if won:
                streak = max(0, streak) + 1
            else:
                streak = min(0, streak) - 1
            streaks.append(streak)
        match_rounds['ct_streak'] = streaks
        
        # Recent performance (last 5 rounds)
        match_rounds['recent_perf'] = (
            match_rounds['ct_won'].rolling(5, min_periods=1).mean() - 0.5
        ) * 2
        
        # Calculate probabilities
        prob_df = calc.calculate_match_win_probability(match_rounds)
        prob_df['match_id'] = match_id
        all_probabilities.append(prob_df)
    
    all_probs_df = pd.concat(all_probabilities, ignore_index=True)
    
    # Visualize sample matches
    print("\nGenerating visualizations...")
    
    # Single match detailed view
    sample_match = all_probs_df['match_id'].iloc[0]
    sample_df = all_probs_df[all_probs_df['match_id'] == sample_match]
    plot_single_match_probability(sample_df, sample_match, 'win_probability_single.png')
    
    # Multiple matches comparison
    sample_matches = {}
    for match_id in all_probs_df['match_id'].unique()[:3]:
        sample_matches[match_id] = all_probs_df[all_probs_df['match_id'] == match_id]
    
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
            plot_single_match_probability(comeback_df, comeback_match, 'comeback_example.png')
    
    # Statistical summary
    print("\n" + "="*50)
    print("PROBABILITY STATISTICS")
    print("="*50)
    
    # Average probability swing per match
    for match_id in all_probs_df['match_id'].unique():
        match_df = all_probs_df[all_probs_df['match_id'] == match_id]
        prob_range = match_df['ct_win_prob'].max() - match_df['ct_win_prob'].min()
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
