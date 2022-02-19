import re
from nba_api.stats.endpoints import commonplayerinfo
import time
from tqdm import tqdm
import pandas as pd

stats = pd.read_csv('playercareerstats.csv').sort_values(by=['PLAYER_ID'])

def further_attributes(player_id):
    api_call = commonplayerinfo.CommonPlayerInfo(player_id).get_data_frames()[0]
    weight_player = int(api_call['WEIGHT'].iloc[0]) * 0.45359237 # pounds to kg
    height_player_raw = re.findall("\d", api_call['HEIGHT'].iloc[0])
    height_player = (int(height_player_raw[0]) * 12 + int(height_player_raw[1])) * 2.54 # inches to cm
    exp_player = api_call['SEASON_EXP'].iloc[0]
    return {'player_id': player_id, 'Height (cm)': height_player, 'Weight (kg)': weight_player, 'Experience': exp_player}


results = []
for p_id in tqdm(list(stats['PLAYER_ID'].unique())):
    time.sleep(2)
    res_player = further_attributes(p_id)
    results.append(res_player)


results_df = pd.DataFrame(results)
results_df.to_csv('data/rec_engine/further_attributes.csv', index = False)