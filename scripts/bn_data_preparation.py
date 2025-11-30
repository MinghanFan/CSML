"""
Bayesian Network Data Preparation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import mutual_info_score

# Configuration
DATA_DIR = Path("clean_dataset")
OUTPUT_DIR = Path("bn_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CS2 Domain Knowledge Constants
PISTOL_ROUNDS = [1, 13]
REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = 24

# Economy thresholds (total team equipment value)
ECO_THRESHOLD = 5000  # < 5000 = eco
SEMI_ECO_THRESHOLD = 10000  # 5000-10000 = semi-eco
SEMI_BUY_THRESHOLD = 20000  # 10000-20000 = semi-buy
# > 20000 = full buy


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load rounds and round_players data, merge, and engineer basic features.
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load base data
    rounds = pd.read_csv(DATA_DIR / "rounds.csv")
    round_players = pd.read_csv(DATA_DIR / "round_players.csv")
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    
    print(f"Loaded {len(rounds):,} rounds from {rounds['match_id'].nunique()} matches")
    print(f"Loaded {len(round_players):,} player-round records")
    
    # Aggregate player stats by team
    round_players['survived'] = (
        round_players['survived'].astype(str).str.lower()
        .map({'true': 1, 'false': 0})
        .fillna(0)
    )
    
    team_stats = (
        round_players.groupby(['match_id', 'round_num', 'team'])
        .agg(
            kills=('kills', 'sum'),
            deaths=('deaths', 'sum'),
            assists=('assists', 'sum'),
            damage=('damage', 'sum'),
            headshots=('headshots', 'sum'),
            survivors=('survived', 'sum'),
        )
        .reset_index()
    )
    
    # Pivot to get CT and T columns
    team_stats_wide = team_stats.pivot(
        index=['match_id', 'round_num'],
        columns='team'
    )
    team_stats_wide.columns = [f"{stat}_{team}" for stat, team in team_stats_wide.columns]
    team_stats_wide = team_stats_wide.reset_index()
    
    # Merge with rounds
    df = rounds.merge(team_stats_wide, on=['match_id', 'round_num'], how='left')
    df = df.merge(matches[['match_id', 'map_name']], on='match_id', how='left')
    
    # Sort by match and round
    df = df.sort_values(['match_id', 'round_num']).reset_index(drop=True)
    
    print(f"Created dataset with {len(df):,} rounds and {len(df.columns)} columns")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features with CS2 domain knowledge.
    
    Key features:
    - Round type detection (pistol, eco, force buy, full buy)
    - Equipment advantages
    - Momentum indicators
    - Score pressure
    - Map/side context
    """
    print("\n" + "="*80)
    print("ENGINEERING FEATURES")
    print("="*80)
    
    df = df.copy()
    
    # ==========================================
    # TARGET VARIABLE
    # ==========================================
    df['ct_win'] = (df['round_winner'] == 'ct').astype(int)
    
    # ==========================================
    # ROUND CONTEXT
    # ==========================================
    df['is_pistol_round'] = df['round_num'].isin(PISTOL_ROUNDS).astype(int)
    df['is_first_half'] = (df['round_num'] <= REGULATION_HALF_ROUNDS).astype(int)
    df['is_second_half'] = (
        (df['round_num'] > REGULATION_HALF_ROUNDS) & 
        (df['round_num'] <= REGULATION_TOTAL_ROUNDS)
    ).astype(int)
    df['is_overtime'] = (df['round_num'] > REGULATION_TOTAL_ROUNDS).astype(int)
    
    # Round number normalized (for temporal effects)
    df['round_num_normalized'] = df['round_num'] / 30.0
    
    # ==========================================
    # SCORE TRACKING (Available at round start)
    # ==========================================
    df['ct_score'] = df.groupby('match_id')['ct_win'].cumsum().shift(1).fillna(0).astype(int)
    df['t_score'] = df.groupby('match_id', group_keys=False).apply(
        lambda x: (1 - x['ct_win']).cumsum().shift(1).fillna(0)
    ).astype(int)
    
    df['score_diff'] = df['ct_score'] - df['t_score']
    df['score_total'] = df['ct_score'] + df['t_score']
    df['ct_score_pct'] = np.where(
        df['score_total'] > 0,
        df['ct_score'] / df['score_total'],
        0.5
    )
    
    # ==========================================
    # MOMENTUM (Win streaks)
    # ==========================================
    df['ct_won_prev'] = df.groupby('match_id')['ct_win'].shift(1).fillna(0.5)
    
    # Win streaks (positive = CT winning, negative = T winning)
    for match_id, group in df.groupby('match_id'):
        streak = 0
        streaks = []
        for won in group['ct_win']:
            if won:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
            streaks.append(streak)
        
        # Shift to get streak entering this round
        df.loc[group.index, 'ct_win_streak'] = pd.Series(streaks).shift(1).fillna(0).values
    
    # ==========================================
    # ECONOMY (Equipment values and round types)
    # ==========================================
    
    # IMPORTANT: Equipment values at round start (NO DATA LEAKAGE)
    # The ct_equipment_value and t_equipment_value from the rounds table
    # represent the STARTING equipment (measured at freeze time end).
    # This is BEFORE the round outcome, so it's a valid predictor.
    # 
    # In CS2, at the start of each round (freeze time):
    # - Players buy weapons/utility
    # - Freeze time ends (round officially starts)  
    # - Equipment values are recorded ← THIS is what we have
    # - Round plays out → outcome
    
    df['ct_equipment'] = df['ct_equipment_value'].fillna(0)
    df['t_equipment'] = df['t_equipment_value'].fillna(0)
    df['equipment_diff'] = df['ct_equipment'] - df['t_equipment']
    
    # Round type detection based on equipment at round start
    def classify_round_type(equip_value):
        """Classify round economy type based on total team equipment value"""
        if equip_value < ECO_THRESHOLD:
            return 'eco'
        elif equip_value < SEMI_ECO_THRESHOLD:
            return 'semi_eco'
        elif equip_value < SEMI_BUY_THRESHOLD:
            return 'semi_buy'
        else:
            return 'full_buy'
    
    df['ct_round_type'] = df['ct_equipment'].apply(classify_round_type)
    df['t_round_type'] = df['t_equipment'].apply(classify_round_type)
    
    # Buy situation (both teams' economy state at round start)
    df['buy_situation'] = df['ct_round_type'] + '_vs_' + df['t_round_type']
    
    # ==========================================
    # MAP-SPECIFIC FEATURES
    # ==========================================
    
    # Calculate empirical CT win rate per map (will be used for prior)
    map_ct_rates = df.groupby('map_name')['ct_win'].mean()
    df['map_ct_winrate'] = df['map_name'].map(map_ct_rates)
    
    # ==========================================
    # HISTORICAL PERFORMANCE (From completed rounds)
    # ==========================================
    
    # These are calculated from PAST rounds only (no data leakage)
    df['kills_diff_actual'] = df['kills_ct'] - df['kills_t']
    df['damage_diff_actual'] = df['damage_ct'] - df['damage_t']
    df['survivors_diff_actual'] = df['survivors_ct'] - df['survivors_t']
    
    # Lag features (shift by 1+ to avoid leakage)
    for lag in [1, 2, 3]:
        df[f'kills_diff_lag{lag}'] = df.groupby('match_id')['kills_diff_actual'].shift(lag)
        df[f'damage_diff_lag{lag}'] = df.groupby('match_id')['damage_diff_actual'].shift(lag)
    
    # Rolling averages of past performance
    for window in [3, 5]:
        df[f'kills_diff_roll{window}'] = (
            df.groupby('match_id')['kills_diff_actual']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f'damage_diff_roll{window}'] = (
            df.groupby('match_id')['damage_diff_actual']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    print(f"Engineered {len(df.columns)} total features")
    print(f"\nRound type distribution (previous round):")
    print(df['buy_situation'].value_counts().head(10))
    
    print(f"\nPistol rounds: {df['is_pistol_round'].sum()}")
    print(f"Overtime rounds: {df['is_overtime'].sum()}")
    
    return df


def analyze_feature_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze continuous feature distributions to determine optimal discretization bins.
    """
    print("\n" + "="*80)
    print("ANALYZING FEATURE DISTRIBUTIONS")
    print("="*80)
    
    features_to_discretize = {
        'equipment_diff': 'Equipment Advantage (CT - T)',
        'ct_win_streak': 'Momentum (Win Streak)',
        'score_diff': 'Score Pressure (CT - T)',
        'map_ct_winrate': 'Map CT Win Rate',
        'kills_diff_roll3': 'Recent Performance (3-round avg kills diff)',
        'damage_diff_roll3': 'Recent Performance (3-round avg damage diff)',
    }
    
    discretization_recommendations = {}
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (feature, label) in enumerate(features_to_discretize.items()):
        ax = axes[idx]
        
        # Get feature data
        data = df[feature].dropna()
        
        # Calculate percentiles for binning
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        bin_edges = np.percentile(data, percentiles)
        
        # Plot distribution
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label}\n(n={len(data):,})')
        
        # Add percentile lines
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for i, (p, edge) in enumerate(zip(percentiles[1:-1], bin_edges[1:-1])):
            ax.axvline(edge, color=colors[i], linestyle='--', alpha=0.5, 
                      label=f'p{p}={edge:.1f}')
        
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Calculate correlation with target
        corr = df[[feature, 'ct_win']].corr().iloc[0, 1]
        ax.text(0.02, 0.98, f'Corr with CT win: {corr:.3f}', 
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Store recommendations
        discretization_recommendations[feature] = {
            'percentiles': dict(zip(percentiles, bin_edges.tolist())),
            'correlation_with_target': corr,
            'mean': float(data.mean()),
            'std': float(data.std())
        }
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=200, bbox_inches='tight')
    print(f"Saved feature_distributions.png")
    
    # Print recommendations
    print("\nDiscretization Recommendations:")
    print("-" * 80)
    for feature, rec in discretization_recommendations.items():
        print(f"\n{feature}:")
        print(f"  Correlation with CT win: {rec['correlation_with_target']:.3f}")
        print(f"  Recommended bins (percentile-based):")
        for percentile, value in rec['percentiles'].items():
            print(f"    p{percentile:3d} = {value:8.1f}")
    
    return discretization_recommendations


def create_discretized_features(
    df: pd.DataFrame, 
    recommendations: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create discretized versions of continuous features based on analysis.
    """
    print("\n" + "="*80)
    print("CREATING DISCRETIZED FEATURES")
    print("="*80)
    
    df_disc = df.copy()
    
    # Equipment advantage (5 bins: strong advantages for both sides + even)
    df_disc['equip_advantage'] = pd.cut(
        df['equipment_diff'],
        bins=[-np.inf, -4000, -1500, 1500, 4000, np.inf],
        labels=['T_strong', 'T_moderate', 'even', 'CT_moderate', 'CT_strong']
    )
    
    # Momentum (5 bins: streaks for both sides)
    df_disc['momentum'] = pd.cut(
        df['ct_win_streak'],
        bins=[-np.inf, -2.5, -0.5, 0.5, 2.5, np.inf],
        labels=['T_streak', 'T_slight', 'neutral', 'CT_slight', 'CT_streak']
    )
    
    # Score pressure (3 bins: close, moderate lead, blowout)
    df_disc['score_pressure'] = pd.cut(
        df['score_diff'].abs(),
        bins=[0, 2, 5, np.inf],
        labels=['close', 'moderate', 'blowout']
    )
    
    # Map bias (3 bins: T-favored, balanced, CT-favored)
    df_disc['map_side_bias'] = pd.cut(
        df['map_ct_winrate'],
        bins=[0, 0.47, 0.53, 1.0],
        labels=['T_favored', 'balanced', 'CT_favored']
    )
    
    # Recent performance (3 bins: T better, even, CT better)
    df_disc['recent_performance'] = pd.cut(
        df['kills_diff_roll3'],
        bins=[-np.inf, -2, 2, np.inf],
        labels=['T_performing', 'even', 'CT_performing']
    )
    
    # Round phase (3 bins: first half, second half, overtime)
    df_disc['round_phase'] = pd.cut(
        df['round_num'],
        bins=[0, 12, 24, np.inf],
        labels=['first_half', 'second_half', 'overtime']
    )
    
    # Buy situation (already categorical from buy_situation)
    # Simplify to major categories
    def simplify_buy_situation(situation):
        """Map detailed buy situations to main categories"""
        if pd.isna(situation):
            return np.nan
        
        # Major advantages (eco vs full_buy)
        if 'full_buy_vs_eco' in situation or 'eco_vs_full_buy' in situation:
            return 'major_advantage'
        
        # Both teams full buy
        elif 'full_buy_vs_full_buy' in situation:
            return 'both_full'
        
        # Both teams eco
        elif 'eco_vs_eco' in situation:
            return 'both_eco'
        
        # Semi situations (one team has partial equipment)
        elif 'semi' in situation:
            return 'semi_situation'
        
        # other
        else:
            return np.nan
    
    df_disc['buy_phase'] = df['buy_situation'].apply(simplify_buy_situation)
    
    # Special round types (binary)
    df_disc['is_pistol'] = df['is_pistol_round'].astype(str).map({
        '1': 'pistol', '0': 'not_pistol'
    })
    
    # Target variable
    df_disc['outcome'] = df['ct_win'].map({0: 'T_win', 1: 'CT_win'})
    
    # Print distribution of discretized features
    print("\nDiscretized Feature Distributions:")
    print("-" * 80)
    
    discrete_features = [
        'equip_advantage', 'momentum', 'score_pressure', 
        'map_side_bias', 'recent_performance', 'round_phase',
        'buy_phase', 'is_pistol', 'outcome'
    ]
    
    for feature in discrete_features:
        print(f"\n{feature}:")
        counts = df_disc[feature].value_counts()
        for category, count in counts.items():
            pct = count / len(df_disc) * 100
            print(f"  {category:20s}: {count:6,} ({pct:5.1f}%)")
    
    return df_disc


def test_conditional_independence(df_disc: pd.DataFrame) -> pd.DataFrame:
    """
    Test conditional independence between feature pairs to validate BN structure.
    
    Uses Chi-square test for independence between categorical variables.
    """
    print("\n" + "="*80)
    print("TESTING CONDITIONAL INDEPENDENCE")
    print("="*80)
    
    features = [
        'equip_advantage', 'momentum', 'score_pressure',
        'map_side_bias', 'recent_performance', 'round_phase',
        'buy_phase', 'is_pistol'
    ]
    
    target = 'outcome'
    
    # Test 1: Direct association with outcome
    print("\n1. Direct Association with Outcome (Chi-square tests):")
    print("-" * 80)
    
    associations = []
    
    for feature in features:
        # Create contingency table
        contingency = pd.crosstab(df_disc[feature], df_disc[target])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Cramér's V (effect size)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        associations.append({
            'feature': feature,
            'chi2': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'significant': 'YES' if p_value < 0.001 else 'NO'
        })
        
        print(f"{feature:25s}: χ²={chi2:8.1f}, p={p_value:.2e}, "
              f"Cramér's V={cramers_v:.3f} [{associations[-1]['significant']}]")
    
    # Test 2: Pairwise feature dependencies
    print("\n2. Pairwise Feature Dependencies:")
    print("-" * 80)
    
    dependency_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i >= j:
                dependency_matrix.loc[feat1, feat2] = np.nan
                continue
            
            contingency = pd.crosstab(df_disc[feat1], df_disc[feat2])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            dependency_matrix.loc[feat1, feat2] = cramers_v
    
    # Visualize dependency matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # We populated only the upper triangle; mask the lower to reveal values
    mask = np.tril(np.ones_like(dependency_matrix.astype(float), dtype=bool))
    
    sns.heatmap(
        dependency_matrix.astype(float),
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.5,
        ax=ax,
        cbar_kws={'label': "Cramér's V (Association Strength)"}
    )
    
    ax.set_title("Feature Dependency Matrix\n(Higher = Stronger Association)", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_dependencies.png', dpi=200, bbox_inches='tight')
    print(f"Saved feature_dependencies.png")
    
    # Test 3: Conditional independence tests
    print("\n3. Conditional Independence Tests:")
    print("-" * 80)
    print("Testing: X ⊥ Y | Z (is X independent of Y given Z?)")
    
    conditional_tests = [
        ('momentum', 'outcome', 'equip_advantage', 
         "Is momentum independent of outcome given equipment?"),
        ('recent_performance', 'outcome', 'equip_advantage',
         "Is recent performance independent of outcome given equipment?"),
        ('score_pressure', 'momentum', None,
         "Is score pressure independent of momentum?"),
        ('buy_phase', 'equip_advantage', None,
         "Is buy phase independent of equipment advantage?"),
    ]
    
    for X, Y, Z, description in conditional_tests:
        print(f"\n{description}")
        print(f"  Testing: {X} ⊥ {Y}" + (f" | {Z}" if Z else ""))
        
        if Z is None:
            # Unconditional test
            contingency = pd.crosstab(df_disc[X], df_disc[Y])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            print(f"  Result: χ²={chi2:.1f}, p={p_value:.2e}")
            print(f"  Conclusion: {'INDEPENDENT' if p_value > 0.05 else 'DEPENDENT'}")
        else:
            # Conditional test (stratified by Z)
            z_values = df_disc[Z].dropna().unique()
            
            total_chi2 = 0
            total_dof = 0
            
            for z_val in z_values:
                subset = df_disc[df_disc[Z] == z_val]
                
                if len(subset) < 30:  # Skip small subsets
                    continue
                
                contingency = pd.crosstab(subset[X], subset[Y])
                
                # Need at least 2x2 table
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                
                chi2, _, dof, _ = chi2_contingency(contingency)
                total_chi2 += chi2
                total_dof += dof
            
            # Combined test
            from scipy.stats import chi2 as chi2_dist
            p_value = 1 - chi2_dist.cdf(total_chi2, total_dof)
            
            print(f"  Result: χ²={total_chi2:.1f}, df={total_dof}, p={p_value:.2e}")
            print(f"  Conclusion: {'CONDITIONALLY INDEPENDENT' if p_value > 0.05 else 'CONDITIONALLY DEPENDENT'}")
    
    # Save association results
    associations_df = pd.DataFrame(associations).sort_values('cramers_v', ascending=False)
    associations_df.to_csv(OUTPUT_DIR / 'feature_associations.csv', index=False)
    print(f"\nSaved feature_associations.csv")
    
    return associations_df

def analyze_special_scenarios(df: pd.DataFrame, df_disc: pd.DataFrame):
    """
    Analyze pistol rounds, eco vs full buy, and overtime scenarios.
    """
    print("\n" + "="*80)
    print("SPECIAL SCENARIO ANALYSIS")
    print("="*80)
    
    scenarios = {
        'Pistol Rounds': df[df['is_pistol_round'] == 1],
        'Overtime Rounds': df[df['is_overtime'] == 1],
        'CT Full Buy vs T Eco': df[df['buy_situation'] == 'full_buy_vs_eco'],
        'T Full Buy vs CT Eco': df[df['buy_situation'] == 'eco_vs_full_buy'],
        'Both Full Buy': df[df['buy_situation'] == 'full_buy_vs_full_buy'],
        'Both Eco': df[df['buy_situation'] == 'eco_vs_eco'],
        'CT Semi vs T Eco': df[df['buy_situation'].str.contains('semi.*_vs_eco', na=False)],
        'Both Semi Buy': df[df['buy_situation'].str.contains('semi.*_vs_semi', na=False)],
    }
    
    results = []
    
    for scenario_name, scenario_df in scenarios.items():
        if len(scenario_df) == 0:
            continue
        
        ct_winrate = scenario_df['ct_win'].mean()
        n_rounds = len(scenario_df)
        
        results.append({
            'scenario': scenario_name,
            'n_rounds': n_rounds,
            'ct_winrate': ct_winrate,
            't_winrate': 1 - ct_winrate,
            'pct_of_total': n_rounds / len(df) * 100
        })
        
        print(f"\n{scenario_name}:")
        print(f"  Rounds: {n_rounds:,} ({n_rounds/len(df)*100:.1f}% of total)")
        print(f"  CT Win Rate: {ct_winrate:.1%}")
        print(f"  T Win Rate: {1-ct_winrate:.1%}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'scenario_analysis.csv', index=False)
    print(f"\nSaved scenario_analysis.csv")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Win rates by scenario
    scenarios_plot = results_df.sort_values('ct_winrate')
    
    y_pos = np.arange(len(scenarios_plot))
    ax1.barh(y_pos, scenarios_plot['ct_winrate'], alpha=0.7, label='CT Win Rate', color='blue')
    ax1.barh(y_pos, scenarios_plot['t_winrate'], left=scenarios_plot['ct_winrate'], 
            alpha=0.7, label='T Win Rate', color='orange')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(scenarios_plot['scenario'])
    ax1.set_xlabel('Win Rate')
    ax1.set_title('Win Rates by Scenario Type')
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='50% (balanced)')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='x')
    
    # Scenario frequency
    ax2.barh(y_pos, scenarios_plot['n_rounds'], alpha=0.7, color='green')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(scenarios_plot['scenario'])
    ax2.set_xlabel('Number of Rounds')
    ax2.set_title('Scenario Frequency in Dataset')
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scenario_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved scenario_analysis.png")
    
    return results_df


def main():
    """Main execution"""
    
    print("="*80)
    print("BAYESIAN NETWORK DATA PREPARATION - SESSION 1")
    print("="*80)
    
    # Step 1: Load and prepare
    df = load_and_prepare_data()
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Analyze distributions
    recommendations = analyze_feature_distributions(df)
    
    # Step 4: Create discretized features
    df_disc = create_discretized_features(df, recommendations)
    
    # Step 5: Test independence
    associations_df = test_conditional_independence(df_disc)
    
    # Step 6: Analyze special scenarios
    scenario_results = analyze_special_scenarios(df, df_disc)
    
    # Save processed data
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    # Save both continuous and discretized versions
    df.to_csv(OUTPUT_DIR / 'rounds_with_features.csv', index=False)
    print(f"Saved rounds_with_features.csv ({len(df):,} rounds)")
    
    df_disc.to_csv(OUTPUT_DIR / 'rounds_discretized.csv', index=False)
    print(f"Saved rounds_discretized.csv ({len(df_disc):,} rounds)")
    
    # Save recommendations
    with open(OUTPUT_DIR / 'discretization_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved discretization_recommendations.json")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nData prepared for Bayesian Network modeling:")
    print(f"  - Total rounds: {len(df):,}")
    print(f"  - Features engineered: {len(df.columns)}")
    print(f"  - Discretized features: {len([c for c in df_disc.columns if c not in df.columns])}")
    print(f"\nKey findings:")
    print(f"  - Pistol rounds: {df['is_pistol_round'].sum():,} ({df['is_pistol_round'].sum()/len(df)*100:.1f}%)")
    print(f"  - Overtime rounds: {df['is_overtime'].sum():,} ({df['is_overtime'].sum()/len(df)*100:.1f}%)")
    print(f"  - Most predictive feature: {associations_df.iloc[0]['feature']} (Cramér's V = {associations_df.iloc[0]['cramers_v']:.3f})")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
