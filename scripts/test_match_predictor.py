"""
Evaluate the match predictor on actual match data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from match_prediction import MatchPredictor


def load_match_data(data_dir: Path) -> pd.DataFrame:
    """Load match data with team compositions."""
    print("Loading match data...")
    
    mp = pd.read_csv(data_dir / "match_players.csv")
    
    # Try to load cluster data if available
    cluster_path = data_dir / "match_players_with_clusters.csv"
    if cluster_path.exists():
        mp = pd.read_csv(cluster_path)
    
    # Get match outcomes (one row per match with both teams)
    matches = []
    
    for match_id in mp['match_id'].unique():
        match_data = mp[mp['match_id'] == match_id]
        
        # Get teams
        teams = match_data['team'].unique()
        if len(teams) != 2:
            continue
        
        team_a_data = match_data[match_data['team'] == teams[0]]
        team_b_data = match_data[match_data['team'] == teams[1]]
        
        # Get players
        team_a_players = team_a_data['player_name'].tolist()
        team_b_players = team_b_data['player_name'].tolist()
        
        if len(team_a_players) != 5 or len(team_b_players) != 5:
            continue
        
        # Get winner
        team_a_won = team_a_data['won_match'].iloc[0]
        
        matches.append({
            'match_id': match_id,
            'team_a_players': team_a_players,
            'team_b_players': team_b_players,
            'team_a_won': int(team_a_won)
        })
    
    matches_df = pd.DataFrame(matches)
    print(f"  Loaded {len(matches_df)} complete matches")
    
    return matches_df


def evaluate_predictor(
    predictor: MatchPredictor,
    test_matches: pd.DataFrame
) -> dict:
    """Evaluate predictor on test matches."""
    print("\n" + "="*80)
    print("EVALUATING MATCH PREDICTOR")
    print("="*80)
    
    predictions = []
    true_labels = []
    probabilities = []
    
    print(f"\nPredicting {len(test_matches)} test matches...")
    
    for idx, row in test_matches.iterrows():
        try:
            result = predictor.predict_match(
                row['team_a_players'],
                row['team_b_players']
            )
            
            pred = 1 if result['predicted_winner'] == 'Team A' else 0
            prob = result['team_a_win_probability']
            true = row['team_a_won']
            
            predictions.append(pred)
            probabilities.append(prob)
            true_labels.append(true)
            
        except Exception as e:
            print(f"  Warning: Failed to predict match {row['match_id']}: {e}")
            continue
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Test Matches: {len(test_matches)}")
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"AUC:          {auc:.3f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        true_labels, 
        predictions,
        target_names=['Team B Wins', 'Team A Wins']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                 Pred B Wins  Pred A Wins")
    print(f"True B Wins      {cm[0, 0]:11d}  {cm[0, 1]:11d}")
    print(f"True A Wins      {cm[1, 0]:11d}  {cm[1, 1]:11d}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities,
        'confusion_matrix': cm
    }


def plot_results(results: dict, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Pred B Wins', 'Pred A Wins'],
        yticklabels=['True B Wins', 'True A Wins']
    )
    plt.title('Match Prediction Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=200)
    plt.close()
    
    # Probability distribution by outcome
    plt.figure(figsize=(10, 6))
    
    probs = results['probabilities']
    true_labels = results['true_labels']
    
    plt.hist(
        probs[true_labels == 0],
        bins=20,
        alpha=0.5,
        label='Team B Actually Won',
        color='orange'
    )
    plt.hist(
        probs[true_labels == 1],
        bins=20,
        alpha=0.5,
        label='Team A Actually Won',
        color='blue'
    )
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Predicted Team A Win Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distribution.png', dpi=200)
    plt.close()
    
    # Calibration plot (binned)
    plt.figure(figsize=(8, 6))
    
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    
    observed_freq = []
    for i in range(len(bin_centers)):
        mask = bin_indices == i
        if mask.sum() > 0:
            observed_freq.append(true_labels[mask].mean())
        else:
            observed_freq.append(np.nan)
    
    plt.plot(bin_centers, observed_freq, 'o-', label='Observed')
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability (Team A Win)')
    plt.ylabel('Observed Frequency (Team A Wins)')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_plot.png', dpi=200)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def example_predictions(
    predictor: MatchPredictor,
    matches_df: pd.DataFrame,
    n_examples: int = 5
):
    """Show some example predictions."""
    print("\n" + "="*80)
    print(f"EXAMPLE PREDICTIONS")
    print("="*80)
    
    sample = matches_df.sample(n=min(n_examples, len(matches_df)), random_state=42)
    
    for idx, row in sample.iterrows():
        result = predictor.predict_match(
            row['team_a_players'],
            row['team_b_players']
        )
        
        actual_winner = 'Team A' if row['team_a_won'] else 'Team B'
        correct = '[YES]' if result['predicted_winner'] == actual_winner else '[NO]'
        
        print(f"\nMatch {row['match_id']} {correct}")
        print(f"  Team A: {', '.join(row['team_a_players'])}")
        print(f"  Team B: {', '.join(row['team_b_players'])}")
        print(f"  Predicted: {result['predicted_winner']} ({result['confidence']:.1%})")
        print(f"  Actual:    {actual_winner}")


def main():
    data_dir = Path("clean_dataset")
    model_path = Path("match_predictor_eval/match_predictor.pkl")
    output_dir = Path("match_predictor_eval")
    
    # Load match data
    matches_df = load_match_data(data_dir)
    
    # Split into train/test (80/20)
    train_matches, test_matches = train_test_split(
        matches_df,
        test_size=0.2,
        random_state=13
    )
    
    print(f"\nTrain matches: {len(train_matches)}")
    print(f"Test matches:  {len(test_matches)}")
    
    # Create and train predictor
    predictor = MatchPredictor(data_dir=data_dir)
    
    # Check if model exists
    if model_path.exists():
        print(f"\nLoading existing model from {model_path}")
        predictor.load(model_path)
    else:
        print(f"\nTraining new model...")
        mp = pd.read_csv(data_dir / "match_players.csv")
        
        # Try to load cluster data
        cluster_path = data_dir / "match_players_with_clusters.csv"
        if cluster_path.exists():
            mp = pd.read_csv(cluster_path)
        
        # Filter to training matches only
        train_match_ids = train_matches['match_id'].unique()
        mp_train = mp[mp['match_id'].isin(train_match_ids)]
        
        predictor.load_data()
        predictor.train(mp_train)
        predictor.save(model_path)
    
    # Evaluate on test set
    results = evaluate_predictor(predictor, test_matches)
    
    # Show example predictions
    example_predictions(predictor, test_matches, n_examples=5)
    
    # Create plots
    plot_results(results, output_dir)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nModel: {model_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()