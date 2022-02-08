from nba_api.stats.endpoints import playbyplayv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

def stints_to_dummy_simple(data, df_on_court):

    # merge preprocessing data and df_on_court
    data_stints = pd.merge(data, df_on_court)
    
    # seperate the scores
    scores = data_stints[['GAME_ID', 'stint', 'HOME_PTS', 'AWAY_PTS', 'HOME_PM', 'HOME_PM_100']]
    
    # home and away names
    home_names = ['GUARD_1_HOME', 'GUARD_2_HOME', 'FORWARD_1_HOME', 'FORWARD_2_HOME', 'CENTER_1_HOME']
    away_names = ['GUARD_1_AWAY', 'GUARD_2_AWAY', 'FORWARD_1_AWAY', 'FORWARD_2_AWAY', 'CENTER_1_AWAY']
    all_names = ['GUARD_1_HOME', 'GUARD_2_HOME', 'FORWARD_1_HOME', 'FORWARD_2_HOME', 'CENTER_1_HOME', 'GUARD_1_AWAY', 'GUARD_2_AWAY', 'FORWARD_1_AWAY', 'FORWARD_2_AWAY', 'CENTER_1_AWAY']

    # shift into long format, id: game_id & stint - value vars home and away
    tmp = pd.melt(data_stints, id_vars=['GAME_ID', 'stint'], value_vars=all_names)
    
    # get the dummies on the long format
    tmp2 = pd.get_dummies(tmp, columns=['value'], prefix="", prefix_sep="")
    
    # create boolean and factor vector
    bool_away = tmp2['variable'].isin(away_names)
    weight_vec = np.where(bool_away, -1, 1).reshape(-1, 1)
    
    # extract dummy matrix
    tmp_a = np.asarray(tmp2.iloc[:, 3:])
    
    # adjust for home and away
    tmp_b = tmp_a * weight_vec
    
    # replace
    tmp2.iloc[:, 3:] = tmp_b
    
    tmp2.to_csv("./data_dsp/raw_dummy.csv")
    scores.to_csv("./data_dsp/raw_scores.csv")
    
    # merge scores and players
    df = pd.merge(scores, tmp2)
    
    # get series objects to track the players and their "position"
    series_location_player = pd.Series(df.columns[7:].values)
    series_id_player = series_location_player.apply(lambda x: x.replace('.0', ''))
    series_id_player = series_id_player.astype(int)

    return df, series_id_player

def x_stints_to_dummy_simple(data_stints):
    
    # seperate the scores
    scores = data_stints[['GAME_ID', 'stint', 'HOME_PTS', 'AWAY_PTS', 'HOME_PM', 'HOME_PM_100']]
    
    # home and away names
    home_names = ['HOME_1', 'HOME_2', 'HOME_3', 'HOME_4', 'HOME_5']
    away_names = ['AWAY_1', 'AWAY_2', 'AWAY_3', 'AWAY_4', 'AWAY_5']
    all_names = ['HOME_1', 'HOME_2', 'HOME_3', 'HOME_4', 'HOME_5','AWAY_1', 'AWAY_2', 'AWAY_3', 'AWAY_4', 'AWAY_5']
    # shift into long format, id: game_id & stint - value vars home and away
    tmp = pd.melt(data_stints, id_vars=['GAME_ID', 'stint'], value_vars=all_names)
    
    # get the dummies on the long format
    tmp2 = pd.get_dummies(tmp, columns=['value'], prefix="", prefix_sep="")
    
    # mark the away players
    tmp2.loc[tmp2['variable'].isin(away_names)].iloc[:, 3:] = tmp2.loc[tmp2['variable'].isin(away_names)].iloc[:, 3:] * (-1)

    # merge scores and players
    df = pd.merge(scores, tmp2)
    
    # get series objects to track the players and their "position"
    series_location_player = p.Series(df.columns[6:].values)
    series_id_player = series_location_player.apply(lambda x: x.replace('.0', ''))
    series_id_player = series_id_player.astype(int)

    return dummy_pm, series_id_player

def load_data_pbp(season_ids, limit=None, path_schedule="./data_dsp/schedule.csv", path_boxscores="./data_dsp/boxscores.csv"):
       
    # load the data
    df_schedule = pd.read_csv(path_schedule, dtype={'GAME_ID': str})
    df_boxscores = pd.read_csv(path_boxscores, dtype={'GAME_ID':str})

    print(f'Get data for the season_ids: {season_ids}')

    #game_ids = ["0022001074", "0021900001"]
    game_ids = df_schedule[df_schedule['SEASON_ID'].isin(season_ids)]['GAME_ID'].unique()

    if limit is not None:
        game_ids = game_ids[0:limit]

    list_data = list()

    for game in tqdm(game_ids):

        try:
            # pbp call, append
            call = playbyplayv2.PlayByPlayV2(game_id=game, start_period=1, end_period=4)
            data_load = pd.concat(call.get_data_frames())
            list_data.append(data_load)

        except:
            print(f"Skipped the ID: {game}")
            next

        # sleep
        time.sleep(0.75)

    data_load = pd.concat(list_data)
    data_load = data_load[~data_load['GAME_ID'].isna()]

    return data_load
    
def preprocessing_stint_data(data):

    # transform time, get game_time_left in seconds
    data['game_time_s'] = 60*data['PCTIMESTRING'].apply(lambda x: str(x).split(':')).apply(lambda x: x[0]).astype(float) + data['PCTIMESTRING'].apply(lambda x: str(x).split(':')).apply(lambda x: x[-1]).astype(float)
    data['game_time_left'] = (5 - data['PERIOD'])*data['game_time_s']
    data = data[~data['GAME_ID'].isna()]
    data = data[data['NEUTRALDESCRIPTION'].isna()]

    bool_ft = (data['EVENTMSGTYPE'] == 3).to_numpy() # Free-throws
    bool_fgm = (data['EVENTMSGTYPE'] == 1).to_numpy() # Field goals made
    bool_sub = (data['EVENTMSGTYPE'] == 8).to_numpy() # substitutions
    bool_tech = ((data['EVENTMSGTYPE'] == 6) & (data['EVENTMSGACTIONTYPE'].isin([10, 11, 16, 18, 25]))).to_numpy() # technical fouls
    bool_eject_tech = ((data['EVENTMSGTYPE'] == 11) & (data['EVENTMSGACTIONTYPE'] == 1)).to_numpy() # ejct 2nd technical fouls
    bool_ingame_plays = ~(bool_sub | bool_tech | bool_eject_tech)
    bool_away = (data['HOMEDESCRIPTION'].isna()).to_numpy()
    bool_home = (data['VISITORDESCRIPTION'].isna()).to_numpy()

    data.loc[bool_away, 'TEAM_LOCATION'] = "AWAY"
    data.loc[bool_home, 'TEAM_LOCATION'] = "HOME"

    return data, bool_ingame_plays

    import warnings
    from tqdm import tqdm
    warnings.filterwarnings("ignore")

def create_stint(data):
    """ meant to be used for the pbp data of one game
    """
    # count the substitutions 
    counts = pd.DataFrame(data.groupby('EVENTMSGTYPE')['EVENTMSGTYPE'].count())
    
    # create a counter up to the number of subs
    counter_seq = np.arange(1, counts.loc[8,:].values[0] + 1)
    
    # add stint counter to the subs and fill until next observation
    data.loc[data['EVENTMSGTYPE'] == 8, 'stint'] = counter_seq
    data.loc[:, 'stint'] = data['stint'].fillna(method='ffill')
    
    # fill na with zero (first stint)
    data.loc[data['stint'].isna(), :] = 0
    
    data.loc[:, 'stint'] = data['stint'].astype(int)
    
    return data

def create_stint_leg(data):
	# create the stint_marker
        bool_subs = data['EVENTMSGTYPE'] == 8
        data.loc[bool_subs, 'stint_marker'] = np.arange(start=1, stop=np.sum(bool_subs) + 1)

        # create idx array
        inter_id = np.where(~data['stint_marker'].isna())[0] + 1
        last_id = data.shape[0]
        first_id = 0

        # idx
        idx = np.where(~data['stint_marker'].isna())[0] + 1

        list_stint = list()

        for i, ids in tqdm(enumerate(idx)):

            # first iteration
            if i == 0:
                tmp = data.iloc[:idx[0]]

            # in between
            else:
                tmp = data.iloc[idx[i - 1]:idx[i]]

            # append to list
            list_stint.append(tmp)

        # append the last iteration
        stint = data.iloc[idx[-1]:]
        stint['stint'] = i
        list_stint.append(stint)
        print(i)
        # insert the stint counter
        #for i, stint in enumerate(list_stint):
        #    stint['stint'] = i

        return pd.concat(list_stint)
    
def get_score(data):

    bool_score = ((data['SCORE'] != 0) & (data['SCORE'] != "0") & ((~data['SCORE'].isna())))
    
    # split score into seperate columns
    data.loc[bool_score, 'HOME_PTS'] = data[bool_score]['SCORE'].apply(lambda x: str(x).split(' - ')[0]).astype(int)
    data.loc[bool_score, 'AWAY_PTS'] = data[bool_score]['SCORE'].apply(lambda x: str(x).split(' - ')[1]).astype(int)
    
    # correct wrong zeros
    data.loc[data['AWAY_PTS'].isna(), 'HOME_PTS'] = np.nan
        
    data['HOME_PM'] = data['HOME_PTS'] - data['AWAY_PTS']
    data['AWAY_PM'] = data['AWAY_PTS'] - data['HOME_PTS']
    
    data['HOME_PTS'] = data['HOME_PTS'].fillna(method="ffill")
    data['AWAY_PTS'] = data['AWAY_PTS'].fillna(method="ffill")
    data['HOME_PM'] = data['HOME_PM'].fillna(method="ffill")
    data['AWAY_PM'] = data['AWAY_PM'].fillna(method="ffill")

    return data

def estimate_possessions(df):
    # event types
    event_types = {"OTHER" : 0,
                   "FIELD_GOAL_MADE" : 1,
    "FIELD_GOAL_MISSED" : 2,
    "FREE_THROW" : 3,
    "REBOUND" : 4,
    "TURNOVER" : 5,
    "FOUL" : 6,
    "VIOLATION" : 7,
    "SUBSTITUTION" : 8,
    "TIMEOUT" : 9,
    "JUMP_BALL" : 10,
    "EJECTION" : 11,
    "PERIOD_BEGIN" : 12,
    "PERIOD_END" : 13,
    "UNKNOWN" : 18}

    # change keys and values
    event_types = {y:x for x,y in event_types.items()}

    # EVENTMSGTYPE
    df['EVENTMSGTYPE'] = df['EVENTMSGTYPE'].astype(int)
    df['EVENT'] = df['EVENTMSGTYPE'].apply(lambda x: event_types[x])

    # end of possession events
    end_of_poss = ["FIELD_GOAL_MADE", "FIELD_GOAL_MISSED", "FREE_THROW", "TURNOVER", "REBOUND", "PERIOD_END", "FOUL", "VIOLATION"]

    # indicator
    df.loc[df["EVENT"].isin(end_of_poss),'POSS_CHANGE'] = 1
    df.loc[~df["EVENT"].isin(end_of_poss),'POSS_CHANGE'] = 0

    # changes per team, per game, per sting
    poss_changes = pd.DataFrame(df.groupby(['TEAM_LOCATION', 'GAME_ID', 'stint'])['POSS_CHANGE'].sum()).reset_index()
    poss_changes = poss_changes.rename(columns={'TEAM_LOCATION':'TEAM_LOCATION', 'GAME_ID':'GAME_ID', 'stint':'stint', 'POSS_CHANGE':'EST_POSSESSIONS'})

    # transform into wide format
    poss_changes = poss_changes.set_index(['GAME_ID', 'stint', 'TEAM_LOCATION']).EST_POSSESSIONS.unstack().reset_index().rename_axis(None, axis=1).drop(0, axis=1)

    # merge back to main DataFrame
    df = pd.merge(df, poss_changes, how='left')
    df = df.rename(columns={'HOME':'HOME_POSS', 'AWAY':'AWAY_POSS'})
    
    
    return df


def estimate_pm_100(data):
    
    data['HOME_PM_100'] = data['HOME_PM'] / data['HOME_POSS'] * 100
    data['AWAY_PM_100'] = data['AWAY_PM'] / data['AWAY_POSS'] * 100
    
    return data

import time
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoreadvancedv2

def get_roster_and_starters(data):

    list_starters = list()
    list_roster = list()

    print(data.head())

    unique_games = data['GAME_ID'].unique()

    print(unique_games)

    for gameid in tqdm(unique_games):
        print(gameid)

        try:
            call_boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=gameid)
            game = pd.concat(call_boxscore.get_data_frames())
            # get home or away
            print(game['TEAM_ID'].unique())
            away_team_id = game['TEAM_ID'].unique()[0] # maybe replace by table that has TEAM_LOCATION for all games

            game['TEAM_LOCATION'] = np.where(game['TEAM_ID'] == away_team_id, 'AWAY', 'HOME')


            tmp2 = game[~game['START_POSITION'].isna()][['START_POSITION', 'PLAYER_ID', 'GAME_ID', 'TEAM_ID', 'TEAM_LOCATION']]
            tmp2['STARTER'] = np.where((tmp2['START_POSITION'] == ""), False, True)
            tmp2 = tmp2.drop_duplicates()  # just in case

            list_starters.append(tmp2[tmp2['STARTER']])
            list_roster.append(tmp2)

        except:
            next

        time.sleep(0.75)


            #print(f'ID {gameid} got skipped')
            #next

    return pd.concat(list_starters), pd.concat(list_roster)

def get_all_subs(data):
    bool_sub = data['EVENTMSGTYPE'] == 8
    subs = data.loc[bool_sub, ['GAME_ID', 'stint', 'game_time_left', 'TEAM_LOCATION', 'PLAYER1_ID', 'PLAYER2_ID']]
    subs = subs.rename(columns={'PLAYER1_ID':'PLAYER_OUT_ID', 'PLAYER2_ID':'PLAYER_IN_ID'})
    return subs.sort_values(['GAME_ID', 'stint'])

def get_starting_lineup(starters):

    tmp = starters.groupby(['GAME_ID', 'TEAM_ID'])['PLAYER_ID'].unique().reset_index()
    starting_lineup = tmp.explode('PLAYER_ID')
    starting_lineup['stint'] = 0

    return starting_lineup

def get_on_court(starters, subs):
    """ each row 10 players on court for each stint in each game
    """
    
    # create player role vector
    n_games = starters['GAME_ID'].unique().shape[0]
    naming_vector = ['GUARD_1_HOME', 'GUARD_2_HOME', 'FORWARD_1_HOME', 'FORWARD_2_HOME', 'CENTER_1_HOME', 'GUARD_1_AWAY', 'GUARD_2_AWAY', 'FORWARD_1_AWAY', 'FORWARD_2_AWAY', 'CENTER_1_AWAY']
    player_role_vec = naming_vector * n_games

    # insert into starters
    starters = starters.sort_values(['GAME_ID', 'TEAM_LOCATION', 'START_POSITION'], ascending=False)
    starters['PLAYER_ROLE'] = player_role_vec

    # transform to wide format
    df = pd.pivot_table(starters, values='PLAYER_ID', index=['GAME_ID'], columns=['PLAYER_ROLE'], aggfunc=np.sum).reset_index()

    # rename
    naming_vector.append('GAME_ID')
    df = df[naming_vector]

    naming_vector2 = ['GUARD_1_HOME', 'GUARD_2_HOME', 'FORWARD_1_HOME', 'FORWARD_2_HOME', 'CENTER_1_HOME', 'GUARD_1_AWAY', 'GUARD_2_AWAY', 'FORWARD_1_AWAY', 'FORWARD_2_AWAY', 'CENTER_1_AWAY']
    game_ids = df['GAME_ID'].unique()

    list_dfs = list()
    print("Create on court dataset: \n")

    for i, game in enumerate(tqdm(game_ids)):
            
        # get game id and starters
        game_id = df.iloc[i,-1]
        starter_i = df.iloc[i,:-1].values

        # get subs in that game
        subs_game = subs[subs['GAME_ID'] == game]
        tmp = subs_game

        # extract substitutions
        players_out = subs_game['PLAYER_OUT_ID'].values
        players_in = subs_game['PLAYER_IN_ID'].values

        # get stints
        stints_game = subs_game['stint'].values
        list_oncourt = list()

        for s, stint in enumerate(stints_game):

            # first stint: starters on the court
            if stint == 1:
                on_court_s = starter_i

            # append on court
            list_oncourt.append(on_court_s)    

            # replace players
            on_court_s[on_court_s == players_out[s]] = players_in[s]

        # create dataframe for game
        tmp_on_court = pd.DataFrame(np.concatenate(list_oncourt, axis=0).reshape(-1, 10), columns=naming_vector2)
        tmp_on_court['stint'] = stints_game
        tmp_on_court['GAME_ID'] = game

        # collect
        list_dfs.append(tmp_on_court)

    # create df for all games
    df_on_court = pd.concat(list_dfs)

    return df_on_court

from tqdm import tqdm
def x_get_on_court(data, starters, sub):

    # init
    game_ids = data['GAME_ID'].unique()
    list_on_court = list()
    list_game_id = list()
    list_stint = list()

    # loop over all games
    for game_id in tqdm(game_ids):

        # form subsets
        game = data[data['GAME_ID'] == game_id]
        starting_lineup = starters[starters['GAME_ID'] == game_id]
        subs = sub[sub['GAME_ID'] == game_id]

        # prepare stints for the loop
        a = game['stint'].unique()
        tmp = a[~np.isnan(a)]
        last_stint = np.max(tmp)
        stints = tmp - 1

        # get rosters for the game for both teams
        starters_home = starting_lineup[starting_lineup['TEAM_LOCATION'] == "HOME"]['PLAYER_ID'].unique()
        starters_away = starting_lineup[starting_lineup['TEAM_LOCATION'] == "AWAY"]['PLAYER_ID'].unique()

        # get the substitutions
        players_in, players_out = subs['PLAYER_IN_ID'].values, subs['PLAYER_OUT_ID'].values

        # more init
        on_court = np.zeros((stints.shape[0]+1, 10))
        lineup = np.append(starters_home, starters_away)

        # store game id, so lengths match
        list_game_id.append(game_id)

        for i, stint in enumerate(stints):

            j = i+1

            # for first iteration store starting lineup and first sub
            if stint == 0:
                on_court[i, :] = lineup
                lineup[lineup == players_out[i]] = players_in[i] # sub player out
                on_court[j, :] = lineup

            else:
                lineup[lineup == players_out[i]] = players_in[i]
                on_court[j, :] = lineup

            # store and keep track
            list_game_id.append(game_id)
            list_stint.append(stint)

        # store on court formations
        list_on_court.append(on_court)
        list_stint.append(last_stint)

        # numpy format
        a_on_court = np.concatenate(list_on_court)
        a_game_id = np.asarray(list_game_id)
        a_stint = np.asarray(list_stint)

        # create colnames
        col_names = ['stint']
        col_names.extend([f'HOME_{i}' for i in np.arange(start=1, stop=6)])
        col_names.extend([f'AWAY_{i}' for i in np.arange(start=1, stop=6)])
        col_names.extend(['GAME_ID'])

        # transform to dataframe
        df = pd.DataFrame(data=np.concatenate((a_stint.reshape(-1, 1), a_on_court,
                                               a_game_id.reshape(-1, 1)), axis=1), columns=col_names)

        # adjust datatype
        df['stint'] = df['stint'].astype(float)

    return df

def merge_stint_pts(data, court_data, col_score):

    # store cols for merging and selecting
    col_merge = ['GAME_ID', 'stint']
    col_select = ['GAME_ID', 'stint'] # game_time_left
    col_select.extend(col_score)

    # form subset
    data_subset = data[col_select]

    # merge
    df = pd.merge(court_data, data_subset, how='left', on=col_merge)

    # drop duplicates
    df = df.drop_duplicates()

    # impute values for missing scores
    tmp = df.groupby('GAME_ID')[col_score].ffill()
    df[col_score] = tmp

    # fill NA with zero - these are mostly before there was a score?
    df[col_score] = df[col_score].fillna(value=0)

    return df

def stints_to_dummy(data_stints):

    # colnames, create dummy out of player columns
    col_names = [f'HOME_{i}' for i in np.arange(start=1, stop=6)]
    col_names.extend([f'AWAY_{i}' for i in np.arange(start=1, stop=6)])
    data_dummy = pd.get_dummies(data_stints, prefix_sep='-', columns=col_names)

    # get series objects to track the players and their "position"
    series_location_player = pd.Series(data_dummy.columns[8:].values)
    series_position_player = series_location_player.apply(lambda x: x.split('-')[0])
    series_id_player = series_location_player.apply(lambda x: x.split('-')[1])
    series_id_player = series_id_player.apply(lambda x: x.replace('.0', ''))
    series_id_player = series_id_player.astype(int)

    return data_dummy, series_position_player, series_id_player

def estimate_model(data_dummy, ids_start, col_y, model):
    """
    :param ids_start: index of the column where ids start
    """
    #X = data_dummy.iloc[:, ids_start:].values
    #y = data_dummy[col_y].values

    print("Starting model fitting...")

    model.fit(data_dummy.iloc[:, ids_start:].values, data_dummy[col_y].values)

    player_ids = data_dummy.columns[ids_start:].values

    return model, player_ids

def show_scores_player(coef, series_id_player):
    player_ids = series_id_player

    # load player data
    player_data = pd.read_csv("./data_dsp/players_data.csv")
    player_data['PLAYER_ID'] = player_data['id']

    # data array
    a = np.concatenate((player_ids.reshape(-1,1),
                coef.reshape(-1,1)), axis=1)

    # create dataframe and merge
    df_tmp = pd.DataFrame(a, columns=['PLAYER_ID', 'SCORE'])
    df_tmp['PLAYER_ID'] = df_tmp['PLAYER_ID'].astype(str)
    df_tmp['PLAYER_ID'] = df_tmp['PLAYER_ID'].apply(lambda x: x.replace(".0", "")).astype(int)

    # merge the two
    df_result = pd.merge(df_tmp, player_data)

    return df_result.sort_values('SCORE', ascending=False)

def stints_to_dummy_unique(data_stints, col_scores):
    col_base = ['stint', 'GAME_ID', 'game_time_left']
    col_names = [f'HOME_{i}' for i in np.arange(start=1, stop=6)]
    col_names.extend([f'AWAY_{i}' for i in np.arange(start=1, stop=6)])

    data_slice = data_stints.drop(col_names, axis=1)

    # colnames, create dummy out of player columns
    col_names = [f'HOME_{i}' for i in np.arange(start=1, stop=6)]
    col_names.extend([f'AWAY_{i}' for i in np.arange(start=1, stop=6)])
    tmp = pd.melt(data_stints, id_vars=['stint', 'GAME_ID'], value_vars=col_names)
    tmp_dummy = pd.get_dummies(tmp, columns=['value'], prefix="", prefix_sep="")

    data_dummy = pd.merge(data_slice, tmp_dummy, on=['GAME_ID', 'stint'])

    data_dummy = data_dummy.drop('variable', axis=1)

    # get series objects to track the players and their "position"
    series_location_player = pd.Series(data_dummy.columns[8:].values)
    series_id_player = series_location_player.apply(lambda x: x.replace('.0', ''))
    series_id_player = series_id_player.astype(int)

    return data_dummy, series_id_player


def stints_to_dummy_unique_pm(data_stints):
    home_names = ['HOME_1', 'HOME_2', 'HOME_3', 'HOME_4', 'HOME_5']
    away_names = ['AWAY_1', 'AWAY_2', 'AWAY_3', 'AWAY_4', 'AWAY_5']

    data_home = data_stints.drop(away_names, axis=1)
    data_away = data_stints.drop(home_names, axis=1)

    home_long = pd.melt(data_home, id_vars=['stint', 'GAME_ID', 'HOME_PM_100', 'HOME_PM', 'HOME_PTS', 'AWAY_PTS'],
                        value_vars=home_names)
    dummy_home = pd.get_dummies(home_long, columns=['value'], prefix="", prefix_sep="")

    away_long = pd.melt(data_away, id_vars=['stint', 'GAME_ID', 'HOME_PM_100', 'HOME_PM', 'HOME_PTS', 'AWAY_PTS'],
                        value_vars=away_names)
    dummy_away = pd.get_dummies(away_long, columns=['value'], prefix="", prefix_sep="")
    dummy_away.iloc[:, 3:] = dummy_away.iloc[:, 3:] * (-1)

    dummy_pm = pd.concat([dummy_home, dummy_away], join='inner')
    dummy_pm = dummy_pm.drop('variable', axis=1)

    # get series objects to track the players and their "position"
    series_location_player = pd.Series(dummy_pm.columns[6:].values)
    series_id_player = series_location_player.apply(lambda x: x.replace('.0', ''))
    series_id_player = series_id_player.astype(int)

    return dummy_pm, series_id_player
