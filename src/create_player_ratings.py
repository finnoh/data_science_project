import pandas as pd
import numpy as np
import re
from nba_api.stats.endpoints import playbyplayv2

def stint_lineup(starter_array, subs):
    """

    :param starter_array:
    :param subs:
    :return:
    """
    starters = None
    starters = starter_array.copy()

    print(starters)

    store = list()
    store.append(starters)

    out_player = subs['PLAYER1_ID'].tolist()
    in_player = subs['PLAYER2_ID'].tolist()

    for i, _ in enumerate(out_player):
        if i == 0:
            new_lineup = starters

        current_lineup = new_lineup
        new_lineup = np.where(current_lineup == out_player[i], in_player[i], current_lineup)

        store.append(new_lineup)

    mat = np.stack(store, axis=0)
    _, counts = np.unique(mat, axis=1, return_counts=True)

    assert np.all(counts == 1), "Same player multiple times on the court"
    assert mat.shape[1] == 10, "There are not 10 players on the court"

    return (mat)

def get_stint_pts_data(game_id: str):
    """

    :param game_id:
    :return:
    """
    # load the data
    df_schedule = pd.read_csv("./data/season_prediction/schedule.csv", dtype={'GAME_ID': str})
    df_boxscores = pd.read_csv("./data/season_prediction/boxscores.csv", dtype={'GAME_ID': str})

    # pbp call
    call = playbyplayv2.PlayByPlayV2(game_id=game_id, start_period=1, end_period=4)
    data = pd.concat(call.get_data_frames())
    data.drop(data.tail(1).index, inplace=True)

    # transform time
    data['time_real_tmp'] = data['PCTIMESTRING'].apply(lambda x: re.sub(string=x, repl="", pattern=":"))
    data['time_real_tmp'] = np.abs(data['time_real_tmp'].astype(float) * (5 - data['PERIOD'].astype(float)) - 4800)

    data['HOME_PTS'] = data[data['EVENTMSGTYPE'] == 1]['SCORE'].apply(lambda x: x.split(' - ')[0])
    data['AWAY_PTS'] = data[data['EVENTMSGTYPE'] == 1]['SCORE'].apply(lambda x: x.split(' - ')[1])

    # get the starters of the game
    starters = df_boxscores[df_boxscores['GAME_ID'] == 22001074]
    starters = starters[~starters['START_POSITION'].isna()]

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

    for i, ids in enumerate(idx):

        # first iteration
        if i == 0:
            tmp = data.iloc[:idx[0]]

        # in between
        else:
            tmp = data.iloc[idx[i - 1]:idx[i]]

        # append to list
        list_stint.append(tmp)

    # append the last iteration
    list_stint.append(data.iloc[idx[-1]:])

    # insert the stint counter
    for i, stint in enumerate(list_stint):
        stint['stint'] = i

    data = pd.concat(list_stint)
    game_stints = data['stint'].unique()

    # filter scoring events
    data_score = data[data['EVENTMSGTYPE'] == 1]

    # create data_pts_stint, handle stints where there was no scoring
    tmp_pts_stint = data_score.groupby('stint')[['HOME_PTS', 'AWAY_PTS']].max()
    tmp_pts_stint['stint_merge'] = tmp_pts_stint.index
    tmp2_pts_stint = pd.DataFrame(data={'HOME_PTS': None, 'AWAY_PTS': None, 'stint': game_stints})
    tmp_merge = pd.merge(tmp2_pts_stint, tmp_pts_stint, how='left', left_on='stint', right_on='stint_merge',
                         suffixes=("_drop", None))
    data_pts_stint = tmp_merge.drop(['HOME_PTS_drop', 'AWAY_PTS_drop', 'stint_merge'], axis=1)
    data_pts_stint = data_pts_stint[~data_pts_stint['stint'].isna()]

    subs = data.loc[bool_subs, ['PLAYER1_ID', 'PLAYER2_ID', 'stint', 'stint_marker']]  # all substitutions!
    game = df_boxscores[df_boxscores['GAME_ID'] == game_id]  # get the games boxscore
    starter = game[~game['START_POSITION'].isna()]  # get the starters of the game
    starters = starter['PLAYER_ID'].values

    # get home team bool and id of the games home team
    df_schedule['is_home_game'] = df_schedule['MATCHUP'].apply(lambda x: x.find('@') == -1)
    tmp_ht = df_schedule[df_schedule['GAME_ID'] == game_id]
    home_team_id = tmp_ht[tmp_ht['is_home_game']]['TEAM_ID'].values[0]

    print(home_team_id)

    # get the ids of the home teams players
    home_team_players = game[game['TEAM_ID'] == home_team_id]['PLAYER_ID'].values

    store = list()
    store.append(starters)

    out_player = subs['PLAYER1_ID'].tolist()
    in_player = subs['PLAYER2_ID'].tolist()

    for i, _ in enumerate(out_player):

        if i == 0:
            new_lineup = starters

        current_lineup = new_lineup
        new_lineup = np.where(current_lineup == out_player[i], in_player[i], current_lineup)

        store.append(new_lineup)

    mat = np.stack(store, axis=0)
    _, counts = np.unique(mat, axis=1, return_counts=True)

    assert np.all(counts == 1), "Same player multiple times on the court"
    assert mat.shape[1] == 10, "There are not 10 players on the court"

    # create column-names and dataframe
    colnames_lineup = ['HOME_' + str(i) for i in np.arange(start=1, stop=6)]
    colnames_lineup.extend(['AWAY_' + str(i) for i in np.arange(start=1, stop=6)])
    game_lineups = pd.DataFrame(data=mat, columns=colnames_lineup)

    # store stints in vector
    stint = np.arange(game_lineups.shape[0])

    # save as long format
    game_lineups_long = pd.melt(game_lineups)
    game_lineups_long['value'] = game_lineups_long['value'].astype(str)

    # transform to numpy matrix
    game_lineups = game_lineups.to_numpy()
    game_lineups_shape = game_lineups.shape

    # get unique players used in the game
    player_used = np.unique(game_lineups_long['value'])
    n_player_used = player_used.shape[0]

    # create vector indicating home team players
    cond_ht_players = np.isin(player_used.astype(float), home_team_players)

    # init ohe matrix
    ohe = np.zeros((game_lineups_shape[0], n_player_used))

    # loop over the players used and create dummy variable for each one per stint
    for i, player in enumerate(player_used):

        # home player gets a 1
        if cond_ht_players[i]:
            ohe[:, i] = np.sum((game_lineups == float(player)), axis=1)

        # away player gets a -1
        else:
            ohe[:, i] = np.sum((game_lineups == float(player)), axis=1) * (-1)

    assert np.all(np.abs(ohe).sum(axis=1) == 10), "In some stint, there are not 10 players on the court"
    # assert np.all(np.abs(ohe).max() == 1), "Players have been counted multiple times"

    # dirty hotfix - delete later
    ohe[ohe > 1] = 1
    ohe[ohe < -1] = -1

    # transform to dataframe
    data_player_stint = pd.DataFrame(ohe, columns=player_used)

    # add additional data
    data_player_stint['stint'] = data_player_stint.index
    data_player_stint['GAME_ID'] = game_id

    # impute missing values
    data_pts = data_pts_stint.fillna(method='ffill')

    data_pts[data_pts['HOME_PTS'].isna()] = 0
    data_pts[data_pts['AWAY_PTS'].isna()] = 0

    data_pts['HOME_PTS'] = data_pts['HOME_PTS'].astype(float)
    data_pts['AWAY_PTS'] = data_pts['AWAY_PTS'].astype(float)

    # plus minus
    data_pts['HOME_PLUS_MINUS'] = data_pts['HOME_PTS'] - data_pts['AWAY_PTS']
    data_pts['AWAY_PLUS_MINUS'] = data_pts['AWAY_PTS'] - data_pts['HOME_PTS']

    # delta of plus-minus
    data_pts['HOME_PM_DIFF'] = data_pts['HOME_PLUS_MINUS'].diff()
    data_pts['AWAY_PM_DIFF'] = data_pts['AWAY_PLUS_MINUS'].diff()
    data_pts[data_pts['HOME_PM_DIFF'].isna()] = 0
    data_pts[data_pts['AWAY_PM_DIFF'].isna()] = 0

    data_stint = pd.merge(data_player_stint, data_pts)

    return data_stint
