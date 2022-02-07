from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, commonplayerinfo
import time
import pandas as pd
from tqdm import tqdm

teams = {'2020':['MIL', 'PHX', 'LAC', 'ATL'], '2019':['LAL', 'MIA', 'DEN', 'BOS'], '2018':['TOR', 'GSW', 'MIL', 'POR'], '2017': ['GSW', 'CLE', 'HOU', 'BOS']} 

list_boxscores = list()
for season_str in tqdm(teams.keys()):
    teams_interesting = set(teams[season_str])
    call_season = leaguegamelog.LeagueGameLog(season = season_str, season_type_all_star = ['Playoffs'])
    season = pd.concat(call_season.get_data_frames())
    game_ids = season['GAME_ID'].unique()
    for i, game in enumerate(tqdm(game_ids)):
        time.sleep(0.600)
        call_boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game)
        boxscore_df = pd.concat(call_boxscore.get_data_frames())
        teams_df = set(boxscore_df['TEAM_ABBREVIATION'].unique())
        intersection_teams = teams_df & teams_interesting
        if len(intersection_teams) > 0: # one of the two participating teams is interesting for us
            for team in intersection_teams:
                boxscore_df_team = boxscore_df[boxscore_df['TEAM_ABBREVIATION'] == team]
                boxscore_df_team['SEASON'] = season_str # or using insert
                list_boxscores.append(boxscore_df_team)
        if i % 10 == 0:
            time.sleep(5)
    
df_boxscores = pd.concat(list_boxscores)
df_boxscores.rename(columns={'Unnamed: 0': 'player_game_index'}, inplace=True)
df_boxscores = df_boxscores.dropna(subset=['PLAYER_ID'])
df_boxscores['PLAYER_ID'] = df_boxscores['PLAYER_ID'].apply(lambda x: str(x)[:-2])
df_boxscores['MIN'] = df_boxscores['MIN'].apply(lambda x: int(x[-2:]) + 60*int(x[:-3]) if x is not None else 0)
df_boxscores.to_csv('playoffs_boxscores.csv', index = False)


players_info = []
for i, player in tqdm(enumerate(df_boxscores['PLAYER_ID'].unique())):
    time.sleep(0.600)
    player_response = commonplayerinfo.CommonPlayerInfo(player).get_data_frames()[0]
    player_pos = player_response['POSITION'][0][0]
    player_name = player_response['DISPLAY_FIRST_LAST'][0]
    players_info.append({'PLAYER_ID': player, 'NAME': player_name, 'POS': player_pos})
    if i % 10 == 0:
        time.sleep(5)
players_info = pd.DataFrame(players_info)
players_info.to_csv('playoffs_players.csv', index = False)

'''
def get_playoff_boxscores(season_str: str):
    call_season = leaguegamelog.LeagueGameLog(season = season_str, season_type_all_star = ['Playoffs'])
    season = pd.concat(call_season.get_data_frames())
    game_ids = season['GAME_ID'].unique()
    list_boxscores = list()

    for i, game in enumerate(tqdm(game_ids)):
        time.sleep(0.600)
        call_boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game)
        list_boxscores.append(pd.concat(call_boxscore.get_data_frames()))
        if i % 10 == 0:
            time.sleep(5)
    df_boxscores = pd.concat(list_boxscores)
    df_boxscores.rename(columns={'Unnamed: 0': 'player_game_index'}, inplace=True)
    return df_boxscores

players_info = []

playoffs_2017 = get_playoff_boxscores('2017')
playoffs_2017 = playoffs_2017.dropna(subset=['PLAYER_ID'])
playoffs_2017['PLAYER_ID'] = playoffs_2017['PLAYER_ID'].apply(lambda x: str(x)[:-2])
playoffs_2017['MIN'] = playoffs_2017['MIN'].apply(lambda x: int(x[-2:]) + 60*int(x[:-3]) if x is not None else 0)
playoffs_2017.to_csv('playoffs_17_18.csv', index = False)

for i, player in tqdm(enumerate(playoffs_2017['PLAYER_ID'].unique())):
    time.sleep(0.600)
    player_response = commonplayerinfo.CommonPlayerInfo(player).get_data_frames()[0]
    player_pos = player_response['POSITION'][0][0]
    player_name = player_response['DISPLAY_FIRST_LAST'][0]
    players_info.append({'PLAYER_ID': player, 'NAME': player_name, 'POS': player_pos})
    if i % 10 == 0:
        time.sleep(5)


playoffs_2018 = get_playoff_boxscores('2018')
playoffs_2018 = playoffs_2018.dropna(subset=['PLAYER_ID'])
playoffs_2018['PLAYER_ID'] = playoffs_2018['PLAYER_ID'].apply(lambda x: str(x)[:-2])
playoffs_2018['MIN'] = playoffs_2018['MIN'].apply(lambda x: int(x[-2:]) + 60*int(x[:-3]) if x is not None else 0)
playoffs_2018['PLAYER_ID'] = playoffs_2018['PLAYER_ID'].astype(str)
playoffs_2018.to_csv('playoffs_18_19.csv', index = False)

for i, player in tqdm(enumerate(playoffs_2018['PLAYER_ID'].unique())):
    time.sleep(0.600)
    player_response = commonplayerinfo.CommonPlayerInfo(player).get_data_frames()[0]
    player_pos = player_response['POSITION'][0][0]
    player_name = player_response['DISPLAY_FIRST_LAST'][0]
    players_info.append({'PLAYER_ID': player, 'NAME': player_name, 'POS': player_pos})
    if i % 10 == 0:
        time.sleep(5)


playoffs_2019 = get_playoff_boxscores('2019')
playoffs_2019 = playoffs_2019.dropna(subset=['PLAYER_ID'])
playoffs_2019['PLAYER_ID'] = playoffs_2019['PLAYER_ID'].apply(lambda x: str(x)[:-2])
playoffs_2019['MIN'] = playoffs_2019['MIN'].apply(lambda x: int(x[-2:]) + 60*int(x[:-3]) if x is not None else 0)
playoffs_2019['PLAYER_ID'] = playoffs_2019['PLAYER_ID'].astype(str)
playoffs_2019.to_csv('playoffs_19_20.csv', index = False)

for i, player in tqdm(enumerate(playoffs_2019['PLAYER_ID'].unique())):
    time.sleep(0.600)
    player_response = commonplayerinfo.CommonPlayerInfo(player).get_data_frames()[0]
    player_pos = player_response['POSITION'][0][0]
    player_name = player_response['DISPLAY_FIRST_LAST'][0]
    players_info.append({'PLAYER_ID': player, 'NAME': player_name, 'POS': player_pos})
    if i % 10 == 0:
        time.sleep(5)


playoffs_2020 = get_playoff_boxscores('2020')
playoffs_2020 = playoffs_2020.dropna(subset=['PLAYER_ID'])
playoffs_2020['PLAYER_ID'] = playoffs_2020['PLAYER_ID'].apply(lambda x: str(x)[:-2])
playoffs_2020['MIN'] = playoffs_2020['MIN'].apply(lambda x: int(x[-2:]) + 60*int(x[:-3]) if x is not None else 0)
playoffs_2020['PLAYER_ID'] = playoffs_2020['PLAYER_ID'].astype(str)
playoffs_2020.to_csv('playoffs_20_21.csv', index = False)

for i, player in tqdm(enumerate(playoffs_2020['PLAYER_ID'].unique())):
    time.sleep(0.600)
    player_response = commonplayerinfo.CommonPlayerInfo(player).get_data_frames()[0]
    player_pos = player_response['POSITION'][0][0]
    player_name = player_response['DISPLAY_FIRST_LAST'][0]
    players_info.append({'PLAYER_ID': player, 'NAME': player_name, 'POS': player_pos})
    if i % 10 == 0:
        time.sleep(5)


player_info = pd.DataFrame(players_info)
players_info.to_csv('playoffs_player.csv', index = False)
'''