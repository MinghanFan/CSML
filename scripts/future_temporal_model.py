"""
Temporal Boosted Tree Model for Round Prediction
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    log_loss, 
    brier_score_loss
)
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')

REGULATION_HALF_ROUNDS = 12
REGULATION_TOTAL_ROUNDS = REGULATION_HALF_ROUNDS * 2
OVERTIME_HALF_ROUNDS = 3
OVERTIME_BLOCK_ROUNDS = OVERTIME_HALF_ROUNDS * 2


class RoundFeatureEngineer:
    """
    Creates temporal features for round-level prediction.
    Strict causality: features for round t use only info from rounds <= t-1
    """
    
    def __init__(self, lag_rounds=[1, 2, 3, 5, 7, 10], window_sizes=[3, 5]):
        self.lag_rounds = lag_rounds
        self.window_sizes = window_sizes
    
    def extract_base_features(self, rounds_df, round_players_df, kills_df, damages_df, 
                             grenades_df, bomb_df, match_id):
        """Extract base features for each round from raw data"""
        
        # Get rounds for this match
        match_rounds = rounds_df[rounds_df['match_id'] == match_id].sort_values('round_num')
        
        if len(match_rounds) == 0:
            return pd.DataFrame()
        
        features_list = []
        
        for idx, round_row in match_rounds.iterrows():
            round_num = round_row['round_num']
            
            # Get round data
            round_kills = kills_df[
                (kills_df['match_id'] == match_id) & 
                (kills_df['round_num'] == round_num)
            ]
            
            round_damages = damages_df[
                (damages_df['match_id'] == match_id) & 
                (damages_df['round_num'] == round_num)
            ]
            
            round_grenades = grenades_df[
                (grenades_df['match_id'] == match_id) & 
                (grenades_df['round_num'] == round_num)
            ]
            
            round_bomb = bomb_df[
                (bomb_df['match_id'] == match_id) & 
                (bomb_df['round_num'] == round_num)
            ]
            
            round_player_stats = round_players_df[
                (round_players_df['match_id'] == match_id) & 
                (round_players_df['round_num'] == round_num)
            ]
            
            # Aggregate stats per team
            ct_stats = round_player_stats[round_player_stats['team'] == 'ct']
            t_stats = round_player_stats[round_player_stats['team'] == 't']
            
            # Basic round info
            features = {
                'match_id': match_id,
                'round_num': round_num,
                'round_winner': 1 if round_row['round_winner'] == 'ct' else 0,  # CT wins = 1
                
                # Equipment differential (from rounds table)
                'ct_equipment_value': round_row['ct_equipment_value'],
                't_equipment_value': round_row['t_equipment_value'],
                'equipment_diff': round_row['ct_equipment_value'] - round_row['t_equipment_value'],
                
                # Players alive at end
                'ct_alive_end': round_row['ct_players_alive_end'],
                't_alive_end': round_row['t_players_alive_end'],
                'alive_diff_end': round_row['ct_players_alive_end'] - round_row['t_players_alive_end'],
                
                # Bomb events
                'bomb_planted': 1 if round_row['bomb_planted'] else 0,
                'bomb_site_a': 1 if round_row.get('bomb_site') == 'A' else 0,
                'bomb_site_b': 1 if round_row.get('bomb_site') == 'B' else 0,
            }
            
            # Kill-based features
            ct_kills = round_kills[round_kills['attacker_side'] == 'ct']
            t_kills = round_kills[round_kills['attacker_side'] == 't']
            
            features['ct_kills'] = len(ct_kills)
            features['t_kills'] = len(t_kills)
            features['kill_diff'] = len(ct_kills) - len(t_kills)
            
            # First kill (opening duel)
            if len(round_kills) > 0:
                first_kill = round_kills.iloc[0]
                features['ct_got_first_kill'] = 1 if first_kill['attacker_side'] == 'ct' else 0
                features['t_got_first_kill'] = 1 if first_kill['attacker_side'] == 't' else 0
            else:
                features['ct_got_first_kill'] = 0
                features['t_got_first_kill'] = 0
            
            # Headshot kills
            features['ct_headshots'] = len(ct_kills[ct_kills['hitgroup'] == 'head'])
            features['t_headshots'] = len(t_kills[t_kills['hitgroup'] == 'head'])
            
            # Multi-kills (2+ kills by same player)
            ct_multikills = ct_kills.groupby('attacker_steamid').size()
            t_multikills = t_kills.groupby('attacker_steamid').size()
            features['ct_multikill_rounds'] = sum(ct_multikills >= 2)
            features['t_multikill_rounds'] = sum(t_multikills >= 2)
            
            # Damage features
            ct_damage = round_damages[
                (round_damages['attacker_side'] == 'ct') & 
                (round_damages['victim_side'] == 't')
            ]
            t_damage = round_damages[
                (round_damages['attacker_side'] == 't') & 
                (round_damages['victim_side'] == 'ct')
            ]
            
            features['ct_damage'] = ct_damage['dmg_health_real'].sum()
            features['t_damage'] = t_damage['dmg_health_real'].sum()
            features['damage_diff'] = features['ct_damage'] - features['t_damage']
            
            # Utility usage (grenades)
            ct_nades = round_grenades[round_grenades['thrower_side'] == 'ct']
            t_nades = round_grenades[round_grenades['thrower_side'] == 't']
            
            # Count different grenade types
            for nade_type in ['smoke', 'flash', 'hegrenade', 'molotov', 'incgrenade']:
                ct_count = len(ct_nades[ct_nades['grenade_type'].str.contains(nade_type, case=False, na=False)])
                t_count = len(t_nades[t_nades['grenade_type'].str.contains(nade_type, case=False, na=False)])
                features[f'ct_{nade_type}'] = ct_count
                features[f't_{nade_type}'] = t_count
                features[f'{nade_type}_diff'] = ct_count - t_count
            
            # Utility damage
            utility_weapons = ['hegrenade', 'molotov', 'inferno', 'incgrenade']
            ct_util_dmg = round_damages[
                (round_damages['attacker_side'] == 'ct') & 
                (round_damages['weapon'].isin(utility_weapons))
            ]
            t_util_dmg = round_damages[
                (round_damages['attacker_side'] == 't') & 
                (round_damages['weapon'].isin(utility_weapons))
            ]
            
            features['ct_utility_damage'] = ct_util_dmg['dmg_health_real'].sum()
            features['t_utility_damage'] = t_util_dmg['dmg_health_real'].sum()
            features['utility_damage_diff'] = features['ct_utility_damage'] - features['t_utility_damage']
            
            # Weapon type proxy (from kills)
            weapon_categories = {
                'awp': ['awp'],
                'rifle': ['ak47', 'm4a1', 'm4a1_silencer', 'famas', 'galil', 'sg553', 'aug'],
                'smg': ['mp9', 'mp7', 'mp5', 'ump45', 'p90', 'bizon', 'mac10'],
                'pistol': ['usp_silencer', 'glock', 'p250', 'fiveseven', 'tec9', 'deagle', 'elite']
            }
            
            for weapon_cat, weapons in weapon_categories.items():
                ct_weapon_kills = len(ct_kills[ct_kills['weapon'].isin(weapons)])
                t_weapon_kills = len(t_kills[t_kills['weapon'].isin(weapons)])
                features[f'ct_{weapon_cat}_kills'] = ct_weapon_kills
                features[f't_{weapon_cat}_kills'] = t_weapon_kills
                features[f'{weapon_cat}_kill_diff'] = ct_weapon_kills - t_weapon_kills
            
            # Player performance aggregates
            if len(ct_stats) > 0:
                features['ct_avg_damage'] = ct_stats['damage'].mean()
                features['ct_total_damage'] = ct_stats['damage'].sum()
                features['ct_survivors'] = ct_stats['survived'].sum()
            else:
                features['ct_avg_damage'] = 0
                features['ct_total_damage'] = 0
                features['ct_survivors'] = 0
            
            if len(t_stats) > 0:
                features['t_avg_damage'] = t_stats['damage'].mean()
                features['t_total_damage'] = t_stats['damage'].sum()
                features['t_survivors'] = t_stats['survived'].sum()
            else:
                features['t_avg_damage'] = 0
                features['t_total_damage'] = 0
                features['t_survivors'] = 0
            
            features['avg_damage_diff'] = features['ct_avg_damage'] - features['t_avg_damage']
            features['survivor_diff'] = features['ct_survivors'] - features['t_survivors']
            
            # Round context
            features['round_index'] = round_num
            features['is_first_half'] = 1 if round_num <= REGULATION_HALF_ROUNDS else 0
            features['is_second_half'] = 1 if (round_num > REGULATION_HALF_ROUNDS and round_num <= REGULATION_TOTAL_ROUNDS) else 0
            features['is_overtime'] = 1 if round_num > REGULATION_TOTAL_ROUNDS else 0
            features['round_in_half'] = (
                round_num
                if round_num <= REGULATION_HALF_ROUNDS
                else (
                    round_num - REGULATION_HALF_ROUNDS
                    if round_num <= REGULATION_TOTAL_ROUNDS
                    else ((round_num - REGULATION_TOTAL_ROUNDS - 1) % OVERTIME_HALF_ROUNDS) + 1
                )
            )
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def add_momentum_features(self, df):
        """Add momentum and streak features using only past rounds"""
        df = df.sort_values(['match_id', 'round_num'])
        
        # Win/loss streaks
        df['ct_won'] = df['round_winner'].astype(int)
        df['t_won'] = 1 - df['ct_won']
        
        # Calculate streaks
        for team in ['ct', 't']:
            streak_col = f'{team}_streak'
            df[streak_col] = 0
            
            for match_id in df['match_id'].unique():
                match_mask = df['match_id'] == match_id
                match_df = df[match_mask].copy()
                
                streak = 0
                streaks = []
                for won in match_df[f'{team}_won']:
                    if won:
                        streak += 1
                    else:
                        streak = 0
                    streaks.append(streak)
                
                df.loc[match_mask, streak_col] = streaks
        
        # Align streak context to the start of each round
        for team in ['ct', 't']:
            streak_col = f'{team}_streak'
            df[streak_col] = df.groupby('match_id')[streak_col].shift(1).fillna(0)
        
        # Score differential at round start
        df['ct_score'] = df.groupby('match_id')['ct_won'].cumsum().shift(1).fillna(0).astype(int)
        df['t_score'] = df.groupby('match_id')['t_won'].cumsum().shift(1).fillna(0).astype(int)
        df['score_diff'] = df['ct_score'] - df['t_score']
        
        return df
    
    def add_lag_features(self, df, feature_cols):
        """Add lag features for specified columns"""
        df = df.sort_values(['match_id', 'round_num'])
        
        # Create lag features
        for col in feature_cols:
            for lag in self.lag_rounds:
                lag_col = f'{col}_lag{lag}'
                df[lag_col] = df.groupby('match_id')[col].shift(lag)
        
        # Create rolling window features
        for col in feature_cols:
            for window in self.window_sizes:
                roll_col = f'{col}_roll{window}'
                df[roll_col] = df.groupby('match_id')[col].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                # Shift by 1 to avoid leakage
                df[roll_col] = df.groupby('match_id')[roll_col].shift(1)
        
        # Cumulative features
        cumulative_cols = ['ct_kills', 't_kills', 'ct_damage', 't_damage', 
                          'ct_got_first_kill', 't_got_first_kill']
        
        for col in cumulative_cols:
            if col in df.columns:
                cum_col = f'{col}_cumsum'
                df[cum_col] = df.groupby('match_id')[col].cumsum()
                # Shift by 1 to avoid leakage
                df[cum_col] = df.groupby('match_id')[cum_col].shift(1)
        
        return df
    
    def create_final_features(self, rounds_df, round_players_df, kills_df, damages_df, 
                            grenades_df, bomb_df):
        """Create complete feature set for all matches"""
        
        all_features = []
        
        for match_id in rounds_df['match_id'].unique():
            match_features = self.extract_base_features(
                rounds_df, round_players_df, kills_df, damages_df, 
                grenades_df, bomb_df, match_id
            )
            
            if len(match_features) > 0:
                all_features.append(match_features)
        
        if not all_features:
            return pd.DataFrame()
        
        df = pd.concat(all_features, ignore_index=True)
        
        # Add momentum features
        df = self.add_momentum_features(df)
        
        # Select features for lag calculation
        lag_features = [
            'kill_diff', 'damage_diff', 'equipment_diff',
            'ct_got_first_kill', 't_got_first_kill',
            'ct_multikill_rounds', 't_multikill_rounds',
            'utility_damage_diff', 'smoke_diff', 'flash_diff',
            'ct_streak', 't_streak', 'score_diff'
        ]
        
        # Filter to existing columns
        lag_features = [col for col in lag_features if col in df.columns]
        
        # Add lag features
        df = self.add_lag_features(df, lag_features)
        
        # Keep only pre-round context + historical features to avoid leakage
        safe_base_cols = [
            'match_id', 'round_num', 'round_winner',
            'round_index', 'is_first_half', 'is_second_half',
            'is_overtime', 'round_in_half',
            'ct_score', 't_score', 'score_diff',
            'ct_streak', 't_streak'
        ]
        temporal_cols = [
            col for col in df.columns
            if any(token in col for token in ['lag', 'roll', 'cumsum'])
        ]
        keep_cols = [col for col in safe_base_cols if col in df.columns] + temporal_cols
        df = df[keep_cols]
        
        return df


class TemporalRoundPredictor:
    """
    LightGBM-based temporal model for round prediction with proper validation
    """
    
    def __init__(self, n_folds=5, calibrate=True):
        self.n_folds = n_folds
        self.calibrate = calibrate
        self.model = None
        self.calibrator = None
        self.feature_importance = None
        self.feature_names = None
    
    def prepare_features(self, df):
        """Select and prepare features for modeling"""
        
        # Exclude target and metadata
        exclude_cols = ['match_id', 'round_num', 'round_winner', 'ct_won', 't_won']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove features with too many nulls (first rounds won't have lags)
        null_threshold = 0.7
        keep_cols = []
        for col in feature_cols:
            if df[col].notna().mean() > (1 - null_threshold):
                keep_cols.append(col)
        
        self.feature_names = keep_cols
        return df[keep_cols].fillna(0)
    
    def train(self, X, y, groups, categorical_features=None):
        """Train model with group k-fold cross-validation"""
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': 42,
            'verbosity': -1
        }
        
        # Group K-fold CV
        gkf = GroupKFold(n_splits=self.n_folds)
        
        val_predictions = np.zeros(len(X))
        val_scores = []
        models = []
        
        print(f"Training with {self.n_folds}-fold group cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            models.append(model)
            
            # Predict on validation set
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            val_predictions[val_idx] = val_pred
            
            # Calculate validation scores
            val_score = {
                'fold': fold + 1,
                'logloss': log_loss(y_val, val_pred),
                'auc': roc_auc_score(y_val, val_pred),
                'avg_precision': average_precision_score(y_val, val_pred),
                'brier': brier_score_loss(y_val, val_pred)
            }
            val_scores.append(val_score)
            
            print(f"Fold {fold+1}: LogLoss={val_score['logloss']:.4f}, "
                  f"AUC={val_score['auc']:.4f}, Brier={val_score['brier']:.4f}")
        
        # Average models (ensemble)
        self.models = models
        
        # Get feature importance from first model
        self.feature_importance = models[0].feature_importance(importance_type='gain')
        
        # Calibrate if requested
        if self.calibrate:
            print("\nCalibrating probabilities...")
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(val_predictions, y)
        
        # Store validation predictions and scores
        self.val_predictions = val_predictions
        self.val_scores = pd.DataFrame(val_scores)
        
        print(f"\nMean validation scores:")
        print(f"  LogLoss: {self.val_scores['logloss'].mean():.4f} ± {self.val_scores['logloss'].std():.4f}")
        print(f"  AUC: {self.val_scores['auc'].mean():.4f} ± {self.val_scores['auc'].std():.4f}")
        print(f"  Brier: {self.val_scores['brier'].mean():.4f} ± {self.val_scores['brier'].std():.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities (ensemble average)"""
        predictions = np.zeros(len(X))
        
        for model in self.models:
            predictions += model.predict(X, num_iteration=model.best_iteration)
        
        predictions /= len(self.models)
        
        if self.calibrate and self.calibrator is not None:
            predictions = self.calibrator.predict(predictions)
        
        return predictions
    
    def plot_feature_importance(self, top_n=20):
        """Plot top feature importances"""
        if self.feature_importance is None:
            print("No feature importance available. Train model first.")
            return
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_temporal.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def plot_calibration_curve(self, y_true, y_pred, n_bins=10):
        """Plot reliability diagram (calibration curve)"""
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_pred, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_curve_temporal.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_shap_values(self, X_sample, sample_size=1000):
        """Calculate SHAP values for model interpretation"""
        
        # Sample data if too large
        if len(X_sample) > sample_size:
            X_sample = X_sample.sample(n=sample_size, random_state=42)
        
        # Use first model for SHAP (they should be similar)
        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_temporal.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return shap_values


def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("TEMPORAL BOOSTED TREE MODEL FOR ROUND PREDICTION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data_dir = Path("clean_dataset")
    
    # Load with pandas for easier manipulation
    rounds_df = pd.read_csv(data_dir / "rounds.csv")
    round_players_df = pd.read_csv(data_dir / "round_players.csv")
    
    # Load Polars dataframes and convert to pandas
    kills_df = pl.read_parquet(data_dir / "kills.parquet").to_pandas() if (data_dir / "kills.parquet").exists() else pd.DataFrame()
    damages_df = pl.read_parquet(data_dir / "damages.parquet").to_pandas() if (data_dir / "damages.parquet").exists() else pd.DataFrame()
    grenades_df = pl.read_parquet(data_dir / "grenades.parquet").to_pandas() if (data_dir / "grenades.parquet").exists() else pd.DataFrame()
    bomb_df = pl.read_parquet(data_dir / "bomb.parquet").to_pandas() if (data_dir / "bomb.parquet").exists() else pd.DataFrame()
    
    # If parquet files don't exist, create empty dataframes with expected columns
    if kills_df.empty:
        print("Note: kills.parquet not found. Creating mock data structure...")
        # Create minimal structure for demonstration
        kills_df = pd.DataFrame(columns=['match_id', 'round_num', 'tick', 'attacker_steamid', 
                                        'victim_steamid', 'attacker_side', 'victim_side', 
                                        'weapon', 'hitgroup'])
    
    if damages_df.empty:
        print("Note: damages.parquet not found. Creating mock data structure...")
        damages_df = pd.DataFrame(columns=['match_id', 'round_num', 'attacker_steamid',
                                          'victim_steamid', 'attacker_side', 'victim_side',
                                          'weapon', 'dmg_health_real'])
    
    if grenades_df.empty:
        print("Note: grenades.parquet not found. Creating mock data structure...")
        grenades_df = pd.DataFrame(columns=['match_id', 'round_num', 'thrower_steamid',
                                           'thrower_side', 'grenade_type'])
    
    if bomb_df.empty:
        print("Note: bomb.parquet not found. Creating mock data structure...")
        bomb_df = pd.DataFrame(columns=['match_id', 'round_num', 'event', 'bombsite'])
    
    print(f"Loaded {len(rounds_df)} rounds from {rounds_df['match_id'].nunique()} matches")
    
    # Feature engineering
    print("\nEngineering temporal features...")
    engineer = RoundFeatureEngineer(
        lag_rounds=[1, 2, 3, 5, 7, 10],
        window_sizes=[3, 5]
    )
    
    features_df = engineer.create_final_features(
        rounds_df, round_players_df, kills_df, damages_df, grenades_df, bomb_df
    )
    
    if features_df.empty:
        print("Error: No features could be created. Check your data files.")
        return
    
    print(f"Created {len(features_df.columns)} features for {len(features_df)} rounds")
    
    # Prepare for modeling
    print("\nPreparing data for modeling...")
    
    # Remove first N rounds of each match (no history)
    min_round = 3  # Start predicting from round 3
    features_df = features_df[features_df['round_in_half'] >= min_round]
    
    # Create model
    model = TemporalRoundPredictor(n_folds=5, calibrate=True)
    
    # Prepare features
    X = model.prepare_features(features_df)
    y = features_df['round_winner']
    groups = features_df['match_id']
    
    print(f"Final dataset: {len(X)} rounds, {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    model.train(X, y, groups)
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    importance_df = model.plot_feature_importance(top_n=25)
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Calibration curve
    print("\n" + "="*50)
    print("MODEL CALIBRATION")
    print("="*50)
    
    model.plot_calibration_curve(y, model.val_predictions)
    
    # SHAP analysis
    print("\n" + "="*50)
    print("SHAP ANALYSIS")
    print("="*50)
    
    print("Calculating SHAP values for interpretation...")
    shap_values = model.get_shap_values(X.sample(min(1000, len(X))))
    
    # Save model and features
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    import joblib
    
    # Save model
    joblib.dump(model, 'temporal_round_model.pkl')
    print("Model saved to temporal_round_model.pkl")
    
    # Save feature names
    with open('temporal_features.txt', 'w') as f:
        for feature in model.feature_names:
            f.write(f"{feature}\n")
    print(f"Feature names saved to temporal_features.txt")
    
    # Save processed features
    features_df.to_csv('temporal_features_processed.csv', index=False)
    print("Processed features saved to temporal_features_processed.csv")
    
    # Final summary
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Average LogLoss: {model.val_scores['logloss'].mean():.4f}")
    print(f"Average AUC: {model.val_scores['auc'].mean():.4f}")
    print(f"Average Brier Score: {model.val_scores['brier'].mean():.4f}")
    print(f"Average Precision: {model.val_scores['avg_precision'].mean():.4f}")
    
    print("\n Temporal round prediction model complete!")
    print("Next steps:")
    print("  1. Add GRU/LSTM for momentum modeling")
    print("  2. Create real-time prediction pipeline")
    print("  3. Build win probability visualization")


if __name__ == "__main__":
    main()
