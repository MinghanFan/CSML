"""
Match Prediction
Predict match winner based on team composition using player names
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss


class MatchPredictor:
    """Predict CS2 match outcomes based on team composition."""
    
    def __init__(self, data_dir: Path = Path("clean_dataset")):
        self.data_dir = data_dir
        self.model = None
        self.feature_names = None
        self.player_profiles = None  # Average stats per player
        self.cluster_profiles = None  # Average stats per cluster
        self.player_to_cluster = None  # Mapping of player to cluster
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load match_players and cluster data."""
        print("Loading data...")
        
        # Load base data
        mp = pd.read_csv(self.data_dir / "match_players.csv")
        
        # Try to load cluster data if available
        cluster_path = self.data_dir / "match_players_with_clusters.csv"
        if cluster_path.exists():
            mp = pd.read_csv(cluster_path)
            print(f"  Loaded {len(mp):,} player-match records with clusters")
        else:
            print(f"  Warning: No cluster data found at {cluster_path}")
            print(f"  Will train without cluster features")
            mp['cluster'] = -1  # Placeholder
        
        # Create player profiles (average stats per player across all matches)
        player_features = [
            'kills', 'deaths', 'assists', 'adr',
            'kd_ratio', 'kda_ratio', 'hsp', 'survival_rate',
            'first_kills', 'first_deaths', 'multi_kill_rounds',
            'clutches_won', 'clutches_attempted',
            'flash_assists', 'utility_damage',
            'smokes_thrown', 'flashes_thrown',
            'performance_score'
        ]
        
        available_features = [f for f in player_features if f in mp.columns]
        
        player_profiles = (
            mp.groupby('player_name')[available_features + ['cluster', 'won_match']]
            .agg({
                **{f: 'mean' for f in available_features},
                'cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else -1,
                'won_match': ['sum', 'count']
            })
        )
        
        # Flatten multi-index columns
        player_profiles.columns = [
            f'{col[0]}_{col[1]}' if col[1] else col[0] 
            for col in player_profiles.columns
        ]
        player_profiles = player_profiles.rename(columns={
            'won_match_sum': 'total_wins',
            'won_match_count': 'total_matches'
        })
        player_profiles['win_rate'] = (
            player_profiles['total_wins'] / player_profiles['total_matches']
        )
        
        self.player_profiles = player_profiles
        
        # Create player to cluster mapping
        if 'cluster' in mp.columns:
            self.player_to_cluster = (
                mp.groupby('player_name')['cluster']
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else -1)
                .to_dict()
            )
        
        print(f"  Built profiles for {len(player_profiles)} unique players")
        
        return mp, player_profiles
    
    def build_team_features(
        self, 
        mp: pd.DataFrame,
        use_clusters: bool = True
    ) -> pd.DataFrame:
        """Aggregate individual player stats into team-level features."""
        print("\nBuilding team features...")
        
        # Core aggregated stats
        agg_dict = {
            'adr': 'mean',
            'survival_rate': 'mean',
            'kd_ratio': 'mean',
            'kda_ratio': 'mean',
            'hsp': 'mean',
            'first_kills': 'sum',
            'first_deaths': 'sum',
            'multi_kill_rounds': 'sum',
            'clutches_won': 'sum',
            'clutches_attempted': 'sum',
            'flash_assists': 'sum',
            'utility_damage': 'mean',
            'performance_score': 'mean',
            'smokes_thrown': 'sum',
            'flashes_thrown': 'sum',
            'won_match': 'max'  # Binary target
        }
        
        # Only use features that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in mp.columns}
        
        # Add cluster composition if available
        if use_clusters and 'cluster' in mp.columns and mp['cluster'].nunique() > 1:
            agg_dict['cluster'] = lambda x: x.value_counts().to_dict()
        
        team_data = (
            mp.groupby(['match_id', 'team'])
            .agg(agg_dict)
            .reset_index()
        )
        
        # Expand cluster composition into separate columns
        if 'cluster' in team_data.columns:
            cluster_df = team_data['cluster'].apply(
                lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series()
            ).fillna(0)
            
            # Ensure all cluster columns exist
            for i in range(6):  # Assume max 6 clusters
                col_name = f'cluster_{i}'
                if i not in cluster_df.columns:
                    cluster_df[i] = 0
            
            cluster_df.columns = [f'cluster_{int(c)}' for c in cluster_df.columns]
            team_data = pd.concat([team_data.drop(columns=['cluster']), cluster_df], axis=1)
        
        # Calculate derived metrics
        if 'clutches_attempted' in team_data.columns:
            team_data['clutch_success_rate'] = (
                team_data['clutches_won'] / team_data['clutches_attempted'].replace(0, np.nan)
            ).fillna(0)
        
        if 'first_kills' in team_data.columns and 'first_deaths' in team_data.columns:
            team_data['opening_duel_advantage'] = (
                team_data['first_kills'] - team_data['first_deaths']
            )
        
        print(f"  Created {len(team_data)} team records with {len(team_data.columns)} features")
        
        return team_data
    
    def train(
        self,
        mp: pd.DataFrame,
        n_folds: int = 5,
        random_state: int = 42
    ) -> Dict:
        """Train the match prediction model with cross-validation."""
        print("\n" + "="*80)
        print("TRAINING MATCH PREDICTOR")
        print("="*80)
        
        # Build team features
        use_clusters = 'cluster' in mp.columns and mp['cluster'].nunique() > 1
        team_data = self.build_team_features(mp, use_clusters=use_clusters)
        
        # Separate features and target
        exclude_cols = ['match_id', 'team', 'won_match']
        feature_cols = [c for c in team_data.columns if c not in exclude_cols]
        
        X = team_data[feature_cols].fillna(0)
        y = team_data['won_match'].astype(int)
        groups = team_data['match_id']
        
        self.feature_names = feature_cols
        
        print(f"\nTraining set: {len(X)} teams, {len(feature_cols)} features")
        print(f"Win rate: {y.mean():.1%}")
        print(f"Number of matches: {groups.nunique()}")
        
        # Cross-validation
        gkf = GroupKFold(n_splits=n_folds)
        
        cv_scores = []
        feature_importances = np.zeros(len(feature_cols))
        oof_predictions = np.zeros(len(X))
        
        print(f"\nRunning {n_folds}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=25,
                max_features='sqrt',
                random_state=13
            )
            model.fit(X_train, y_train)
            
            # Predict probabilities
            pred_proba = model.predict_proba(X_val)[:, 1]
            pred = (pred_proba >= 0.5).astype(int)
            
            oof_predictions[val_idx] = pred_proba
            
            # Evaluate
            scores = {
                'fold': fold,
                'accuracy': accuracy_score(y_val, pred),
                'auc': roc_auc_score(y_val, pred_proba),
                'logloss': log_loss(y_val, pred_proba),
                'brier': brier_score_loss(y_val, pred_proba)
            }
            cv_scores.append(scores)
            
            print(f"  Fold {fold}: Acc={scores['accuracy']:.3f}, "
                  f"AUC={scores['auc']:.3f}, LogLoss={scores['logloss']:.3f}")
            
            # Accumulate feature importance
            feature_importances += model.feature_importances_
        
        # Average scores
        cv_scores_df = pd.DataFrame(cv_scores)
        feature_importances /= n_folds
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {cv_scores_df['accuracy'].mean():.3f} ± {cv_scores_df['accuracy'].std():.3f}")
        print(f"AUC:      {cv_scores_df['auc'].mean():.3f} ± {cv_scores_df['auc'].std():.3f}")
        print(f"LogLoss:  {cv_scores_df['logloss'].mean():.3f} ± {cv_scores_df['logloss'].std():.3f}")
        print(f"Brier:    {cv_scores_df['brier'].mean():.3f} ± {cv_scores_df['brier'].std():.3f}")
        
        # Train final model on all data
        print("\nTraining final model on all data...")
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")
        
        return {
            'cv_scores': cv_scores_df,
            'importance': importance_df,
            'oof_predictions': oof_predictions,
            'oof_true': y.values
        }
    
    def get_player_features(self, player_name: str) -> pd.Series:
        """Get average features for a player."""
        if player_name in self.player_profiles.index:
            return self.player_profiles.loc[player_name]
        else:
            # Return average of all players as fallback
            return self.player_profiles.mean()
    
    def predict_match(
        self,
        team_a_players: List[str],
        team_b_players: List[str]
    ) -> Dict:
        """
        Predict match outcome given player names for both teams.
        
        Args:
            team_a_players: List of 5 player names for team A
            team_b_players: List of 5 player names for team B
            
        Returns:
            Dict with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(team_a_players) != 5 or len(team_b_players) != 5:
            raise ValueError("Each team must have exactly 5 players")
        
        # Build features for both teams
        teams_features = []
        
        for team_name, players in [('Team A', team_a_players), ('Team B', team_b_players)]:
            # Get player profiles
            player_stats = [self.get_player_features(p) for p in players]
            
            # Aggregate team stats
            team_features = {}
            
            # Mean stats
            for stat in ['adr', 'survival_rate', 'kd_ratio', 'kda_ratio', 'hsp',
                        'utility_damage', 'performance_score']:
                if stat in player_stats[0].index:
                    team_features[stat] = np.mean([p[stat] for p in player_stats])
            
            # Sum stats
            for stat in ['first_kills', 'first_deaths', 'multi_kill_rounds',
                        'clutches_won', 'clutches_attempted', 'flash_assists',
                        'smokes_thrown', 'flashes_thrown']:
                if stat in player_stats[0].index:
                    team_features[stat] = np.sum([p[stat] for p in player_stats])
            
            # Derived metrics
            if 'clutches_attempted' in team_features and team_features['clutches_attempted'] > 0:
                team_features['clutch_success_rate'] = (
                    team_features['clutches_won'] / team_features['clutches_attempted']
                )
            else:
                team_features['clutch_success_rate'] = 0
            
            if 'first_kills' in team_features and 'first_deaths' in team_features:
                team_features['opening_duel_advantage'] = (
                    team_features['first_kills'] - team_features['first_deaths']
                )
            
            # Cluster composition
            if self.player_to_cluster is not None:
                cluster_counts = {}
                for player in players:
                    cluster = self.player_to_cluster.get(player, -1)
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                
                for i in range(6):
                    team_features[f'cluster_{i}'] = cluster_counts.get(i, 0)
            
            teams_features.append(team_features)
        
        # Create DataFrames with proper feature alignment
        team_a_df = pd.DataFrame([teams_features[0]])
        team_b_df = pd.DataFrame([teams_features[1]])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in team_a_df.columns:
                team_a_df[feature] = 0
            if feature not in team_b_df.columns:
                team_b_df[feature] = 0
        
        # Reorder columns to match training
        team_a_df = team_a_df[self.feature_names]
        team_b_df = team_b_df[self.feature_names]
        
        # Predict
        team_a_win_prob = self.model.predict_proba(team_a_df)[0, 1]
        team_b_win_prob = self.model.predict_proba(team_b_df)[0, 1]
        
        # Normalize probabilities (they should be complements, but just in case)
        total = team_a_win_prob + team_b_win_prob
        team_a_win_prob_normalized = team_a_win_prob / total
        team_b_win_prob_normalized = team_b_win_prob / total
        
        return {
            'team_a_players': team_a_players,
            'team_b_players': team_b_players,
            'team_a_win_probability': team_a_win_prob_normalized,
            'team_b_win_probability': team_b_win_prob_normalized,
            'predicted_winner': 'Team A' if team_a_win_prob_normalized > 0.5 else 'Team B',
            'confidence': max(team_a_win_prob_normalized, team_b_win_prob_normalized)
        }
    
    def save(self, path: Path):
        """Save trained model and player profiles."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'player_profiles': self.player_profiles,
            'player_to_cluster': self.player_to_cluster
        }, path)
        
        print(f"\nModel saved to {path}")
    
    def load(self, path: Path):
        """Load trained model and player profiles."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.player_profiles = data['player_profiles']
        self.player_to_cluster = data.get('player_to_cluster')
        
        print(f"Model loaded from {path}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Players: {len(self.player_profiles)}")


def main():
    parser = argparse.ArgumentParser(description="CS2 Match Predictor")
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'interactive'],
        default='train',
        help='Mode: train model, predict specific match, or interactive prediction'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('clean_dataset'),
        help='Path to clean dataset folder'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('match_predictor_eval/match_predictor.pkl'),
        help='Path to save/load model'
    )
    parser.add_argument(
        '--team-a',
        nargs=5,
        help='Team A player names (5 players)'
    )
    parser.add_argument(
        '--team-b',
        nargs=5,
        help='Team B player names (5 players)'
    )
    
    args = parser.parse_args()
    
    predictor = MatchPredictor(data_dir=args.data_dir)
    
    if args.mode == 'train':
        # Train mode
        mp, player_profiles = predictor.load_data()
        results = predictor.train(mp)
        predictor.save(args.model_path)
        
    elif args.mode == 'predict':
        # Prediction mode
        if not args.team_a or not args.team_b:
            print("Error: --team-a and --team-b required for predict mode")
            return
        
        predictor.load(args.model_path)
        result = predictor.predict_match(args.team_a, args.team_b)
        
        print("\n" + "="*80)
        print("MATCH PREDICTION")
        print("="*80)
        print(f"\nTeam A: {', '.join(result['team_a_players'])}")
        print(f"Team B: {', '.join(result['team_b_players'])}")
        print(f"\nPredicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nWin Probabilities:")
        print(f"  Team A: {result['team_a_win_probability']:.1%}")
        print(f"  Team B: {result['team_b_win_probability']:.1%}")
        
    elif args.mode == 'interactive':
        # Interactive mode
        predictor.load(args.model_path)
        
        print("\n" + "="*80)
        print("INTERACTIVE MATCH PREDICTION")
        print("="*80)
        print("\nEnter player names for both teams (5 players each)")
        
        team_a = []
        print("\nTeam A:")
        for i in range(5):
            player = input(f"  Player {i+1}: ").strip()
            team_a.append(player)
        
        team_b = []
        print("\nTeam B:")
        for i in range(5):
            player = input(f"  Player {i+1}: ").strip()
            team_b.append(player)
        
        result = predictor.predict_match(team_a, team_b)
        
        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"\nTeam A: {', '.join(result['team_a_players'])}")
        print(f"Team B: {', '.join(result['team_b_players'])}")
        print(f"\nPredicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nWin Probabilities:")
        print(f"  Team A: {result['team_a_win_probability']:.1%}")
        print(f"  Team B: {result['team_b_win_probability']:.1%}")


if __name__ == "__main__":
    main()