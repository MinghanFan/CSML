"""
Interactive Match Predictor Demo
Quick demo with example predictions
"""

from pathlib import Path

import random as rand
from match_prediction import MatchPredictor
import pandas as pd


def show_available_players(predictor: MatchPredictor, n: int = 20):
    """Show some available player names from the dataset."""
    if predictor.player_profiles is None:
        print("No player profiles loaded")
        return
    
    # Sort by total matches (most experienced players)
    top_players = (
        predictor.player_profiles
        .sort_values('total_matches', ascending=False)
        .head(n)
    )
    
    print("\nTop players by experience:")
    print(f"{'Player':<30} {'Matches':<10} {'Win Rate':<10} {'ADR':<10} {'KD':<8}")
    print("-" * 70)
    
    for player_name, stats in top_players.iterrows():
        matches = int(stats.get('total_matches', 0))
        win_rate = stats.get('win_rate', 0) * 100
        adr = stats.get('adr_mean', stats.get('adr', 0))
        kd = stats.get('kd_ratio_mean', stats.get('kd_ratio', 0))
        
        print(f"{player_name:<30} {matches:<10} {win_rate:<10.1f} {adr:<10.1f} {kd:<8.2f}")


def predict_example_match(predictor: MatchPredictor):
    """Run an example prediction."""
    # Get top players
    top_players = (
        predictor.player_profiles
        .sort_values('total_matches', ascending=False)
        .head(100)
    )
    
    player_names = top_players.index.tolist()
    
    if len(player_names) < 10:
        print("Not enough players in dataset for example prediction")
        return
    
    # Randomly select 10 players
    rand_int = rand.randint(0, len(player_names) - 10)
    
    # Create two teams
    team_a = player_names[rand_int:rand_int + 5]
    team_b = player_names[rand_int + 5:rand_int + 10]
    
    print("\n" + "="*80)
    print("EXAMPLE MATCH PREDICTION")
    print("="*80)
    
    print(f"\nTeam A: {', '.join(team_a)}")
    print(f"Team B: {', '.join(team_b)}")
    
    result = predictor.predict_match(team_a, team_b)
    
    print(f"\n{'='*60}")
    print("PREDICTION")
    print(f"{'='*60}")
    print(f"Predicted Winner: {result['predicted_winner']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nWin Probabilities:")
    print(f"  Team A: {result['team_a_win_probability']:.1%}")
    print(f"  Team B: {result['team_b_win_probability']:.1%}")


def interactive_mode(predictor: MatchPredictor):
    """Interactive prediction mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MATCH PREDICTION")
    print("="*80)
    
    show_available_players(predictor, n=30)
    
    print("\n" + "="*80)
    print("Enter player names (or press Enter to see example)")
    print("="*80)
    
    # Team A
    print("\nTeam A - Enter 5 player names:")
    team_a = []
    for i in range(5):
        player = input(f"  Player {i+1}: ").strip()
        if player not in predictor.player_profiles.index:
            print(f"    Player '{player}' not found in database. Please try again.")
            return
        elif not player:
            print("\nNo input detected. Running example prediction instead...")
            predict_example_match(predictor)
            return
        team_a.append(player)
    
    # Team B
    print("\nTeam B - Enter 5 player names:")
    team_b = []
    for i in range(5):
        player = input(f"  Player {i+1}: ").strip()
        if player not in predictor.player_profiles.index:
            print(f"    Player '{player}' not found in database. Please try again.")
            return
        elif not player:
            print("\nNo input detected. Running example prediction instead...")
            predict_example_match(predictor)
            return
        team_b.append(player)
    
    # Predict
    print("\n" + "="*80)
    print("PREDICTION")
    print("="*80)
    
    try:
        result = predictor.predict_match(team_a, team_b)
        
        print(f"\nTeam A: {', '.join(result['team_a_players'])}")
        print(f"Team B: {', '.join(result['team_b_players'])}")
        print(f"\nPredicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nWin Probabilities:")
        print(f"  Team A: {result['team_a_win_probability']:.1%}")
        print(f"  Team B: {result['team_b_win_probability']:.1%}")
        
    except Exception as e:
        print(f"\nError making prediction: {e}")


def main():
    data_dir = Path("clean_dataset")
    model_path = Path("match_predictor_eval/match_predictor.pkl")
    
    print("="*80)
    print("CS2 MATCH PREDICTOR DEMO")
    print("="*80)
    
    # Check if model exists
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Training new model...")
        
        predictor = MatchPredictor(data_dir=data_dir)
        mp, _ = predictor.load_data()
        predictor.train(mp)
        predictor.save(model_path)
    else:
        print(f"\nLoading model from {model_path}...")
        predictor = MatchPredictor(data_dir=data_dir)
        predictor.load(model_path)
    
    print(f"\nModel loaded successfully")
    print(f"  Features: {len(predictor.feature_names)}")
    print(f"  Players in database: {len(predictor.player_profiles)}")
    
    # Run interactive mode
    while True:
        print("\n" + "="*80)
        print("OPTIONS")
        print("="*80)
        print("1. Interactive prediction (enter player names)")
        print("2. Example prediction (auto-generate teams)")
        print("3. Show available players")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            interactive_mode(predictor)
        elif choice == '2':
            predict_example_match(predictor)
        elif choice == '3':
            show_available_players(predictor, n=60)
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
