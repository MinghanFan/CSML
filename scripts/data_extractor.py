"""
FIXED Data Extractor for CS2 demos.

FINAL FIXES:
1. Team is "None" for match level (players switch sides)
2. won_match based on actual team membership across the match
3. Grenade counts fixed - count unique throws, not all rows
4. avg_cash_spent fixed
"""

import zipfile
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import hashlib
from datetime import datetime

import polars as pl
from data_schema import CS2DataSchema

class CS2DataExtractor:
    """Extracts clean, structured data from parsed CS2 demos."""
    
    def __init__(self, parsed_demos_folder: Path, output_folder: Path):
        self.parsed_demos_folder = parsed_demos_folder
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.matches_data = []
        self.players_data = defaultdict(lambda: {
            'player_name': None,
            'total_matches': 0,
            'total_kills': 0,
            'total_deaths': 0,
            'total_damage': 0,
            'total_rounds': 0,
        })
        self.match_players_data = []
        self.rounds_data = []
        self.round_players_data = []
    
    def generate_match_id(self, demo_path: Path, header: dict) -> str:
        """Generate unique match ID"""
        unique_string = f"{demo_path.stem}_{header.get('map_name', '')}_{header.get('demo_file_stamp', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def determine_player_team(self, player_id: str, ticks_df: pl.DataFrame, rounds_df: pl.DataFrame) -> tuple:
        """
        FIXED: Determine which actual team a player was on (not side).
        
        Returns: (team_name, starting_side, won_match)
        
        Since we don't have team names in the data:
        - Team 1 = players who started on CT
        - Team 2 = players who started on T
        """
        
        # Get player's side in first round
        first_round = rounds_df['round_num'].min()
        first_round_ticks = ticks_df.filter(
            (pl.col('round_num') == first_round) &
            (pl.col('steamid') == int(player_id))
        )
        
        if len(first_round_ticks) == 0:
            return "None", "unknown", False
        
        starting_side = first_round_ticks['side'][0]
        
        # Determine team based on starting side
        team_name = "Team1" if starting_side == "ct" else "Team2"
        
        # Determine if team won (based on total rounds won across all sides)
        # Count rounds won while on each side
        player_rounds = ticks_df.filter(pl.col('steamid') == int(player_id))
        
        # Get rounds where player was CT
        ct_rounds = player_rounds.filter(pl.col('side') == 'ct')['round_num'].unique()
        # Get rounds where player was T  
        t_rounds = player_rounds.filter(pl.col('side') == 't')['round_num'].unique()
        
        # Count wins for each side
        ct_wins = len(rounds_df.filter(
            (pl.col('round_num').is_in(ct_rounds)) &
            (pl.col('winner') == 'ct')
        ))
        t_wins = len(rounds_df.filter(
            (pl.col('round_num').is_in(t_rounds)) &
            (pl.col('winner') == 't')
        ))
        
        # Team's total wins
        team_wins = ct_wins + t_wins
        
        # Opponent's wins (total rounds - team wins)
        total_rounds = len(rounds_df)
        opponent_wins = total_rounds - team_wins
        
        won_match = team_wins > opponent_wins
        
        return team_name, starting_side, won_match
    
    def extract_all_players(self, data: dict) -> pl.DataFrame:
        """Extract ALL players from multiple sources."""
        
        # Source 1: Players who dealt damage
        damage_attackers = (
            data['damages']
            .select(['attacker_steamid', 'attacker_name', 'attacker_side'])
            .unique()
            .rename({
                'attacker_steamid': 'steamid',
                'attacker_name': 'name',
                'attacker_side': 'side'
            })
        )
        
        # Source 2: Players who received damage
        damage_victims = (
            data['damages']
            .select(['victim_steamid', 'victim_name', 'victim_side'])
            .unique()
            .rename({
                'victim_steamid': 'steamid',
                'victim_name': 'name',
                'victim_side': 'side'
            })
        )
        
        # Source 3: Players from ticks
        tick_players = (
            data['ticks']
            .select(['steamid', 'name', 'side'])
            .unique()
        )
        
        # Combine and deduplicate
        all_players = pl.concat([
            damage_attackers,
            damage_victims,
            tick_players
        ]).unique(subset=['steamid'])
        
        # Filter out invalid
        all_players = all_players.filter(
            (pl.col('steamid') != 0) & 
            (pl.col('steamid').is_not_null())
        )
        
        return all_players
    
    def calculate_utility_damage(self, player_id: str, damages_df: pl.DataFrame) -> int:
        """Calculate utility damage from damage events."""
        utility_weapons = ['hegrenade', 'molotov', 'inferno', 'incgrenade']
        
        utility_dmg = damages_df.filter(
            (pl.col('attacker_steamid') == int(player_id)) &
            (pl.col('weapon').is_in(utility_weapons))
        )
        
        return int(utility_dmg['dmg_health_real'].sum())
    
    def calculate_flash_assists(self, player_id: str, kills_df: pl.DataFrame) -> tuple:
        """Calculate enemies flashed and flash assists."""
        
        # Flash assists: kills where this player flashed the victim
        flash_assists = len(kills_df.filter(
            (pl.col('assister_steamid') == int(player_id)) &
            (pl.col('assistedflash').fill_null(False) == True)
        ))
        
        enemies_flashed = flash_assists
        
        return enemies_flashed, flash_assists
    
    def calculate_clutches(
        self,
        player_id: str,
        player_side: str,
        rounds_df: pl.DataFrame,
        kills_df: pl.DataFrame,
        ticks_df: pl.DataFrame
    ) -> tuple:
        """Calculate clutches correctly."""
        clutches_attempted = 0
        clutches_won = 0
        
        for round_num in rounds_df['round_num']:
            round_row = rounds_df.filter(pl.col('round_num') == round_num)
            if len(round_row) == 0:
                continue
            
            round_winner = round_row['winner'][0]
            
            # Look at kills in this round to find clutch situations
            round_kills = kills_df.filter(pl.col('round_num') == round_num).sort('tick')
            
            if len(round_kills) == 0:
                continue
            
            # For each kill, check if it creates a 1vX situation for our player
            for kill_row in round_kills.iter_rows(named=True):
                kill_tick = kill_row['tick']
                
                # Get alive players at this tick
                tick_data = ticks_df.filter(
                    (pl.col('tick') == kill_tick) &
                    (pl.col('health') > 0)
                )
                
                if len(tick_data) == 0:
                    continue
                
                # Count alive per side
                player_side_alive = tick_data.filter(pl.col('side') == player_side)
                enemy_side_alive = tick_data.filter(pl.col('side') != player_side)
                
                # Check if it's a clutch situation (1vX where X >= 2)
                if len(player_side_alive) == 1 and len(enemy_side_alive) >= 2:
                    # Check if our player is the one alive
                    if int(player_id) in player_side_alive['steamid'].to_list():
                        clutches_attempted += 1
                        
                        # Check if player's side won
                        if player_side == round_winner:
                            clutches_won += 1
                        
                        # Only count once per round
                        break
        
        return clutches_attempted, clutches_won
    
    def count_grenades(self, player_id: str, grenades_df: pl.DataFrame) -> dict:
        """
        FIXED: Count grenade throws correctly.
        
        Problem: Was counting all rows, not unique throws.
        Solution: Count distinct grenade events.
        """
        
        player_grenades = grenades_df.filter(pl.col('thrower_steamid') == int(player_id))
        
        if len(player_grenades) == 0:
            return {
                'smokes': 0,
                'flashes': 0,
                'he_nades': 0,
                'molotovs': 0
            }
        
        # Group by round and tick to count unique throws
        unique_throws = player_grenades.unique(subset=['round_num', 'tick', 'grenade_type'])
        
        # Count each type
        smokes = len(unique_throws.filter(
            pl.col('grenade_type').str.to_lowercase().str.contains('smoke')
        ))
        
        flashes = len(unique_throws.filter(
            pl.col('grenade_type').str.to_lowercase().str.contains('flash')
        ))
        
        he_nades = len(unique_throws.filter(
            pl.col('grenade_type').str.to_lowercase().str.contains('frag|hegrenade|he')
        ))
        
        molotovs = len(unique_throws.filter(
            pl.col('grenade_type').str.to_lowercase().str.contains('molotov|incendiary|inferno')
        ))
        
        return {
            'smokes': smokes,
            'flashes': flashes,
            'he_nades': he_nades,
            'molotovs': molotovs
        }
    
    def extract_match_data(self, demo_path: Path, data: dict) -> dict:
        """Extract match-level data"""
        header = data['header']
        rounds_df = data['rounds']
        
        match_id = self.generate_match_id(demo_path, header)
        
        if len(rounds_df) == 0:
            return None
        
        # Count wins per side
        ct_wins = len(rounds_df.filter(pl.col('winner') == 'ct'))
        t_wins = len(rounds_df.filter(pl.col('winner') == 't'))
        
        match_data = {
            'match_id': match_id,
            'demo_file': demo_path.stem,
            'map_name': header.get('map_name', 'unknown'),
            'date': datetime.now().isoformat(),
            'total_rounds': len(rounds_df),
            'team1_name': 'Team1',  # Team that started CT
            'team2_name': 'Team2',  # Team that started T
            'team1_score': ct_wins,
            'team2_score': t_wins,
            'winner': 'Team1' if ct_wins > t_wins else 'Team2',
            'match_duration_seconds': float(rounds_df['official_end'].max() / 128),
        }
        
        return match_data
    
    def extract_player_match_stats(self, match_id: str, data: dict) -> List[dict]:
        """Extract per-player, per-match statistics."""
        
        kills_df = data['kills']
        damages_df = data['damages']
        ticks_df = data['ticks']
        rounds_df = data['rounds']
        grenades_df = data['grenades']
        
        # Get ALL players
        players = self.extract_all_players(data)
        
        match_players = []
        
        for player_row in players.iter_rows(named=True):
            player_id = str(player_row['steamid'])
            player_name = player_row['name']
            
            # FIXED: Determine actual team and won_match
            team_name, starting_side, won_match = self.determine_player_team(
                player_id, ticks_df, rounds_df
            )
            
            # === KILLS STATS ===
            player_kills = kills_df.filter(pl.col('attacker_steamid') == int(player_id))
            player_deaths = kills_df.filter(pl.col('victim_steamid') == int(player_id))
            player_assists = kills_df.filter(pl.col('assister_steamid') == int(player_id))
            
            total_kills = len(player_kills)
            total_deaths = len(player_deaths)
            total_assists = len(player_assists)
            headshot_kills = len(player_kills.filter(pl.col('hitgroup') == 'head'))
            
            # === DAMAGE STATS ===
            player_damage = damages_df.filter(
                (pl.col('attacker_steamid') == int(player_id)) &
                (pl.col('attacker_side') != pl.col('victim_side'))
            )
            total_damage = int(player_damage['dmg_health_real'].sum())
            
            # Utility damage
            utility_damage = self.calculate_utility_damage(player_id, damages_df)
            
            # Flash assists
            enemies_flashed, flash_assists = self.calculate_flash_assists(player_id, kills_df)
            
            # === SURVIVAL STATS ===
            player_rounds = (
                ticks_df
                .filter(pl.col('steamid') == int(player_id))
                .group_by('round_num')
                .agg(pl.col('health').last())
            )
            rounds_survived = len(player_rounds.filter(pl.col('health') > 0))
            total_rounds = len(player_rounds)
            
            # === OPENING DUELS ===
            first_kills = 0
            first_deaths = 0
            
            for round_num in rounds_df['round_num']:
                round_kills = kills_df.filter(pl.col('round_num') == round_num).sort('tick')
                
                if len(round_kills) > 0:
                    first_kill_tick = round_kills['tick'][0]
                    first_killer = round_kills.filter(pl.col('tick') == first_kill_tick)
                    
                    if len(first_killer) > 0 and first_killer['attacker_steamid'][0] == int(player_id):
                        first_kills += 1
                    
                    if len(first_killer) > 0 and first_killer['victim_steamid'][0] == int(player_id):
                        first_deaths += 1
            
            # === CLUTCHES ===
            clutches_attempted, clutches_won = self.calculate_clutches(
                player_id, starting_side, rounds_df, kills_df, ticks_df
            )
            
            # === UTILITY - FIXED ===
            grenade_counts = self.count_grenades(player_id, grenades_df)
            
            # === CALCULATE METRICS ===
            kd_ratio = total_kills / max(total_deaths, 1)
            kda_ratio = (total_kills + total_assists) / max(total_deaths, 1)
            adr = total_damage / max(total_rounds, 1)
            hsp = (headshot_kills / max(total_kills, 1)) * 100 if total_kills > 0 else 0
            survival_rate = (rounds_survived / max(total_rounds, 1)) * 100
            
            # === MULTI-KILLS ===
            multi_kill_rounds = len(
                player_kills
                .group_by('round_num')
                .agg(pl.count('tick').alias('kills'))
                .filter(pl.col('kills') >= 3)
            )
            
            # === ECONOMY - FIXED ===
            player_economy = ticks_df.filter(pl.col('steamid') == int(player_id))
            
            if len(player_economy) > 0:
                # Average cash spent per round
                cash_per_round = (
                    player_economy
                    .group_by('round_num')
                    .agg(pl.col('cash_spent_this_round').sum().alias('round_cash'))
                )
                avg_cash_spent = float(cash_per_round['round_cash'].mean())
                
                # Average equipment value
                avg_equipment_value = float(player_economy['current_equip_value'].mean())
            else:
                avg_cash_spent = 0.0
                avg_equipment_value = 0.0
            
            match_player_stats = {
                'match_id': match_id,
                'player_id': player_id,
                'player_name': player_name,
                'team': "None",  # FIXED: No team at match level (players switch sides)
                'role': 'unknown',
                
                # Core stats
                'kills': total_kills,
                'deaths': total_deaths,
                'assists': total_assists,
                'headshot_kills': headshot_kills,
                'damage': total_damage,
                'utility_damage': utility_damage,
                'enemies_flashed': enemies_flashed,
                'flash_assists': flash_assists,
                
                # Metrics
                'kd_ratio': kd_ratio,
                'kda_ratio': kda_ratio,
                'adr': adr,
                'hsp': hsp,
                
                # Survival & Impact
                'rounds_survived': rounds_survived,
                'survival_rate': survival_rate,
                'first_kills': first_kills,
                'first_deaths': first_deaths,
                'clutches_attempted': clutches_attempted,
                'clutches_won': clutches_won,
                'multi_kill_rounds': multi_kill_rounds,
                
                # Utility - FIXED
                'smokes_thrown': grenade_counts['smokes'],
                'flashes_thrown': grenade_counts['flashes'],
                'he_thrown': grenade_counts['he_nades'],
                'molotovs_thrown': grenade_counts['molotovs'],
                
                # Economy - FIXED
                'avg_cash_spent_per_round': avg_cash_spent,
                'avg_equipment_value_per_round': avg_equipment_value,
                
                # Performance score (will calculate in next step)
                'performance_score': 0.0,
                
                # Outcome - FIXED: Based on team, not side
                'won_match': won_match,
            }
            
            match_players.append(match_player_stats)
            
            # Update global player stats
            self.players_data[player_id]['player_name'] = player_name
            self.players_data[player_id]['total_matches'] += 1
            self.players_data[player_id]['total_kills'] += total_kills
            self.players_data[player_id]['total_deaths'] += total_deaths
            self.players_data[player_id]['total_damage'] += total_damage
            self.players_data[player_id]['total_rounds'] += total_rounds
        
        return match_players
    
    def extract_round_data(self, match_id: str, data: dict) -> List[dict]:
        """Extract per-round data"""
        rounds_df = data['rounds']
        bomb_df = data['bomb']
        ticks_df = data['ticks']
        
        rounds_data = []
        
        for round_row in rounds_df.iter_rows(named=True):
            round_num = round_row['round_num']
            
            # Bomb events
            round_bomb_events = bomb_df.filter(pl.col('round_num') == round_num)
            bomb_planted = len(round_bomb_events.filter(pl.col('event') == 'plant')) > 0
            bomb_site = None
            if bomb_planted:
                plant_event = round_bomb_events.filter(pl.col('event') == 'plant').head(1)
                if len(plant_event) > 0:
                    bomb_site = plant_event['bombsite'][0]
            
            # Equipment values
            round_start_tick = round_row.get('freeze_end', round_row.get('start'))
            round_equipment = ticks_df.filter(pl.col('tick') == round_start_tick)
            
            ct_equip = int(round_equipment.filter(pl.col('side') == 'ct')['current_equip_value'].sum())
            t_equip = int(round_equipment.filter(pl.col('side') == 't')['current_equip_value'].sum())
            
            # Players alive at end
            round_end_tick = round_row['end']
            round_end_ticks = ticks_df.filter(pl.col('tick') == round_end_tick)
            
            ct_alive = len(round_end_ticks.filter((pl.col('side') == 'ct') & (pl.col('health') > 0)))
            t_alive = len(round_end_ticks.filter((pl.col('side') == 't') & (pl.col('health') > 0)))
            
            round_data = {
                'match_id': match_id,
                'round_num': round_num,
                'round_winner': round_row['winner'],
                'round_end_reason': str(round_row.get('reason', 'unknown')),
                'round_duration': float((round_row['end'] - round_row['start']) / 128),
                'bomb_planted': bomb_planted,
                'bomb_site': bomb_site,
                'ct_equipment_value': ct_equip,
                't_equipment_value': t_equip,
                'ct_players_alive_end': ct_alive,
                't_players_alive_end': t_alive,
            }
            
            rounds_data.append(round_data)
        
        return rounds_data
    
    def extract_round_players_data(self, match_id: str, data: dict) -> List[dict]:
        """Extract per-player, per-round data"""
        kills_df = data['kills']
        damages_df = data['damages']
        ticks_df = data['ticks']
        rounds_df = data['rounds']
        
        round_players_data = []
        
        for round_row in rounds_df.iter_rows(named=True):
            round_num = round_row['round_num']
            
            # Get ALL players in round
            round_ticks = ticks_df.filter(pl.col('round_num') == round_num)
            players_in_round = round_ticks.select(['steamid', 'name', 'side']).unique()
            
            for player_row in players_in_round.iter_rows(named=True):
                player_id = str(player_row['steamid'])
                player_name = player_row['name']
                side = player_row['side']
                
                # Round stats
                round_kills = kills_df.filter(
                    (pl.col('round_num') == round_num) &
                    (pl.col('attacker_steamid') == int(player_id))
                )
                round_deaths = kills_df.filter(
                    (pl.col('round_num') == round_num) &
                    (pl.col('victim_steamid') == int(player_id))
                )
                round_assists = kills_df.filter(
                    (pl.col('round_num') == round_num) &
                    (pl.col('assister_steamid') == int(player_id))
                )
                
                round_damage = damages_df.filter(
                    (pl.col('round_num') == round_num) &
                    (pl.col('attacker_steamid') == int(player_id)) &
                    (pl.col('attacker_side') != pl.col('victim_side'))
                )
                
                # Survival
                player_end_health = (
                    round_ticks
                    .filter(pl.col('steamid') == int(player_id))
                    .sort('tick')
                    .tail(1)['health']
                )
                survived = player_end_health[0] > 0 if len(player_end_health) > 0 else False
                
                # Economy
                player_round_ticks = round_ticks.filter(pl.col('steamid') == int(player_id))
                if len(player_round_ticks) > 0:
                    equip_value = int(player_round_ticks['current_equip_value'].mean())
                    cash_spent = int(player_round_ticks['cash_spent_this_round'].sum())
                    money_start = int(player_round_ticks['balance'].first())
                    money_end = int(player_round_ticks['balance'].last())
                else:
                    equip_value = 0
                    cash_spent = 0
                    money_start = 0
                    money_end = 0
                
                round_player_data = {
                    'match_id': match_id,
                    'round_num': round_num,
                    'player_id': player_id,
                    'player_name': player_name,
                    'team': side,  # At round level, team = side (ct/t)
                    
                    'kills': len(round_kills),
                    'deaths': len(round_deaths),
                    'assists': len(round_assists),
                    'damage': int(round_damage['dmg_health_real'].sum()),
                    'headshots': len(round_kills.filter(pl.col('hitgroup') == 'head')),
                    'survived': survived,
                    
                    'equipment_value': equip_value,
                    'cash_spent': cash_spent,
                    'money_start': money_start,
                    'money_end': money_end,
                    
                    'won_round': side == round_row['winner'],
                }
                
                round_players_data.append(round_player_data)
        
        return round_players_data
    
    def process_single_demo(self, demo_zip: Path) -> dict:
        """Process a single parsed demo"""
        print(f"Processing {demo_zip.name}...")
        
        try:
            with zipfile.ZipFile(demo_zip, 'r') as zipf:
                data = {
                    'header': json.loads(zipf.read('header.json')),
                    'kills': pl.read_parquet(zipf.open('kills.parquet')),
                    'damages': pl.read_parquet(zipf.open('damages.parquet')),
                    'ticks': pl.read_parquet(zipf.open('ticks.parquet')),
                    'rounds': pl.read_parquet(zipf.open('rounds.parquet')),
                    'bomb': pl.read_parquet(zipf.open('bomb.parquet')),
                    'grenades': pl.read_parquet(zipf.open('grenades.parquet')),
                }
            
            # Extract match data
            match_data = self.extract_match_data(demo_zip, data)
            if match_data is None:
                print(f"  ✗ No valid match data")
                return {'success': False, 'error': 'No valid match data'}
            
            match_id = match_data['match_id']
            self.matches_data.append(match_data)
            data['match_data'] = match_data
            
            # Extract player match stats
            match_players = self.extract_player_match_stats(match_id, data)
            self.match_players_data.extend(match_players)
            
            # Extract round data
            rounds_data = self.extract_round_data(match_id, data)
            self.rounds_data.extend(rounds_data)
            
            # Extract round players data
            round_players_data = self.extract_round_players_data(match_id, data)
            self.round_players_data.extend(round_players_data)
            
            print(f"  ✓ Extracted: {len(match_players)} players, {len(rounds_data)} rounds")
            
            return {'success': True, 'match_id': match_id}
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def process_all_demos(self):
        """Process all parsed demos"""
        demo_zips = list(self.parsed_demos_folder.glob("*.zip"))
        
        if len(demo_zips) == 0:
            print(f"✗ No .zip files found in {self.parsed_demos_folder}")
            return []
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {len(demo_zips)} DEMOS")
        print(f"{'='*80}\n")
        
        results = []
        for demo_zip in demo_zips:
            result = self.process_single_demo(demo_zip)
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE: {successful}/{len(demo_zips)} successful")
        print(f"{'='*80}")
        
        return results
    
    def create_dataframes(self) -> Dict[str, pl.DataFrame]:
        """Convert collected data to DataFrames"""
        print("\nCreating DataFrames...")
        
        dataframes = {
            'matches': pl.DataFrame(self.matches_data, schema=CS2DataSchema.matches_schema),
            'players': self._create_players_df(),
            'match_players': pl.DataFrame(self.match_players_data, schema=CS2DataSchema.match_players_schema),
            'rounds': pl.DataFrame(self.rounds_data, schema=CS2DataSchema.rounds_schema),
            'round_players': pl.DataFrame(self.round_players_data, schema=CS2DataSchema.round_players_schema),
        }
        
        print(f"  ✓ Matches: {len(dataframes['matches'])} rows")
        print(f"  ✓ Players: {len(dataframes['players'])} rows")
        print(f"  ✓ Match Players: {len(dataframes['match_players'])} rows")
        print(f"  ✓ Rounds: {len(dataframes['rounds'])} rows")
        print(f"  ✓ Round Players: {len(dataframes['round_players'])} rows")
        
        return dataframes
    
    def _create_players_df(self) -> pl.DataFrame:
        """Create players DataFrame"""
        players_list = []
        
        for player_id, stats in self.players_data.items():
            players_list.append({
                'player_id': player_id,
                'player_name': stats['player_name'],
                'total_matches': stats['total_matches'],
                'primary_role': 'unknown',
                'avg_rank': 0.0,
            })
        
        return pl.DataFrame(players_list, schema=CS2DataSchema.players_schema)
    
    def save_dataframes(self, dataframes: Dict[str, pl.DataFrame]):
        """Save all DataFrames"""
        print("\nSaving DataFrames...")
        
        for name, df in dataframes.items():
            parquet_path = self.output_folder / f"{name}.parquet"
            df.write_parquet(parquet_path)
            print(f"  ✓ Saved {name}.parquet")
            
            csv_path = self.output_folder / f"{name}.csv"
            df.write_csv(csv_path)
            print(f"  ✓ Saved {name}.csv")
        
        print(f"\n✓ All data saved to {self.output_folder}")
    
    def generate_summary(self, dataframes: Dict[str, pl.DataFrame]) -> dict:
        """Generate summary statistics"""
        matches_df = dataframes['matches']
        players_df = dataframes['players']
        match_players_df = dataframes['match_players']
        
        summary = {
            'dataset_statistics': {
                'total_matches': len(matches_df),
                'total_players': len(players_df),
                'total_rounds': dataframes['rounds'].shape[0],
                'avg_rounds_per_match': float(matches_df['total_rounds'].mean()),
                'total_player_performances': len(match_players_df),
            },
            'map_distribution': (
                matches_df
                .group_by('map_name')
                .agg(pl.count('match_id').alias('count'))
                .sort('count', descending=True)
                .to_dicts()
            ),
            'performance_statistics': {
                'avg_kills_per_match': float(match_players_df['kills'].mean()),
                'avg_deaths_per_match': float(match_players_df['deaths'].mean()),
                'avg_adr': float(match_players_df['adr'].mean()),
                'avg_kd_ratio': float(match_players_df['kd_ratio'].mean()),
                'avg_survival_rate': float(match_players_df['survival_rate'].mean()),
                'avg_utility_damage': float(match_players_df['utility_damage'].mean()),
                'avg_flash_assists': float(match_players_df['flash_assists'].mean()),
                'total_clutches_won': int(match_players_df['clutches_won'].sum()),
                'avg_smokes_per_match': float(match_players_df['smokes_thrown'].mean()),
                'avg_flashes_per_match': float(match_players_df['flashes_thrown'].mean()),
                'avg_clutch_success_rate': float(
                    match_players_df.filter(pl.col('clutches_attempted') > 0)
                    .select((pl.col('clutches_won') / pl.col('clutches_attempted')).mean())
                    .item() * 100 if len(match_players_df.filter(pl.col('clutches_attempted') > 0)) > 0 else 0
                ),
            },
        }
        
        return summary
    
    def save_summary(self, summary: dict):
        """Save summary"""
        summary_path = self.output_folder / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"\nDataset Statistics:")
        for key, value in summary['dataset_statistics'].items():
            print(f"  {key}: {value}")
        
        print(f"\nPerformance Statistics:")
        for key, value in summary['performance_statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

print("✓ Data extractor loaded")