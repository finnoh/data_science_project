import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

player_data = pd.read_csv('players_data.csv')
nba_2k = pd.read_csv('data/rec_engine/nba2k_ratings.csv')

# match players from NBA 2k rating and the NBA api to get unique IDs
matched_player_names = []
player_names = [list(nba_2k['player'])[j] for j in range(nba_2k.shape[0])]
for i in range(player_data.shape[0]):
    player_name = list(player_data['player_names'])[i]
    try:
        matched_player_names.append(list(nba_2k[nba_2k['player'] == player_name]['player'])[0])
        player_names.remove(player_name)
        
    except IndexError:
        matched_player_names.append(0)

lost_players = []

for i in range(len(matched_player_names)):
    if matched_player_names[i] == 0:
        player_name = list(player_data['player_names'])[i] #list(nba_2k['player'])[i]
        fuzz_scores = np.array([fuzz.ratio(player_name, player_names[j]) for j in range(len(player_names))])
        max_ind = np.argmax(fuzz_scores)
        if fuzz_scores[max_ind] > 69: #70
            matched_player_name = player_names[max_ind]
            print(f"{player_name} matched to {matched_player_name}")
            matched_player_names[i] = list(nba_2k[nba_2k['player'] == matched_player_name]['player'])[0]
            player_names.remove(matched_player_name)
            print(fuzz_scores[max_ind])
        else:
            #player_ids.append(0)
            print(f"{player_name} NOT matched to {player_names[max_ind]}")
            lost_players.append(player_name)

print('Still available from 2k ratings:', player_names)
print('Not assigned NBA API players:', lost_players, len(lost_players))

# Wrong matches:
# Alize Johnson matched to Tyler Johnson
# Devon Dotson matched to Damyean Dotson

ratings_matched = []
for i in range(player_data.shape[0]):
    p_name = list(player_data['player_names'])[i]
    matched_p_name = matched_player_names[i]

    if (matched_p_name == 0) or (matched_p_name == 'Tyler Johnson') or (matched_p_name == 'Damyean Dotson'):
        dict_p = {'player': p_name, 'total': 0, 'inside': 0, 'outside': 0, 'playmaking': 0, 'athleticism': 0, 'defending': 0, 'rebounding': 0}

    else:
        df_p = nba_2k[nba_2k['player'] == matched_p_name].to_dict('list')
        dict_p = {'player': p_name, 'total': df_p['total'][0], 'inside': df_p['inside'][0], 'outside': df_p['outside'][0], 'playmaking': df_p['playmaking'][0], 'athleticism': df_p['athleticism'][0], 'defending': df_p['defending'][0], 'rebounding': df_p['rebounding'][0]}
    
    ratings_matched.append(dict_p)


ratings_df = pd.DataFrame(ratings_matched)

print(ratings_df.head())