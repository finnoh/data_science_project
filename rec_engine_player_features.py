import re
from nba_api.stats.endpoints import commonplayerinfo
import time
from tqdm import tqdm
import pandas as pd

stats = pd.read_csv('playercareerstats.csv').sort_values(by=['PLAYER_ID'])
playoff_stats = pd.read_csv('data/rec_engine/playoffs_boxscores.csv')

def further_attributes(player_id, season, minus_exp):
    api_call = commonplayerinfo.CommonPlayerInfo(player_id).get_data_frames()[0]
    weight_player = int(api_call['WEIGHT'].iloc[0]) * 0.45359237 # pounds to kg
    height_player_raw = re.findall("[0-9]+", api_call['HEIGHT'].iloc[0])
    height_player = (int(height_player_raw[0]) * 12 + int(height_player_raw[1])) * 2.54 # inches to cm
    exp_player = api_call['SEASON_EXP'].iloc[0] - minus_exp
    return {'player_id': player_id, 'Height (cm)': height_player, 'Weight (kg)': weight_player, 'Experience': exp_player, 'Season': str(season)}


results = []

teams = {2020:['MIL', 'PHX', 'LAC', 'ATL'], 2019:['LAL', 'MIA', 'DEN', 'BOS'], 2018:['TOR', 'GSW', 'MIL', 'POR'], 2017: ['GSW', 'CLE', 'HOU', 'BOS']} 
seasons = [2020, 2019, 2018, 2017, 2016]

for n, season in enumerate(list(teams.keys())):
    teams_season = teams[season]
    for t in tqdm(teams_season):
        df_team = playoff_stats[(playoff_stats['TEAM_ABBREVIATION'] == t) & (playoff_stats['SEASON'] == season)]
        players = list(df_team['PLAYER_ID'].unique())
        for p_id in players:
            time.sleep(1)
            res_player = further_attributes(p_id, seasons[seasons.index(season) + 1], n+2) # want value before the champion seasons
            results.append(res_player)


for p_id in tqdm(list(stats['PLAYER_ID'].unique())):
    time.sleep(1)
    res_player = further_attributes(p_id, '2020', 1)
    results.append(res_player)

results_df = pd.DataFrame(results)
results_df.to_csv('data/rec_engine/further_attributes.csv', index = False)