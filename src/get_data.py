import pandas as pd
import numpy as np
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from PIL import Image
import requests
from tqdm import tqdm
from os import listdir
import time
import matplotlib.pyplot as plt
import seaborn as sns

def get_boxscores_per_season(season_id: str):
    """
    :param season_id: str, season id of the season you want to query
    :return df_boxscores: pd.DataFrame, contains all boxscores of a season
    """
    # get the season schedule
    call_season = leaguegamelog.LeagueGameLog(season=season_id)
    season = pd.concat(call_season.get_data_frames())

    # get that season's game ids and setup list as storage object
    game_ids = season['GAME_ID'].unique()
    list_boxscores = list()

    # loop over game ids and store boxscores
    for i, game in enumerate(tqdm(game_ids)):
        call_boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game)
        list_boxscores.append(pd.concat(call_boxscore.get_data_frames()))

        # take a break
        time.sleep(1)

    # concatenate to dataframe
    df_boxscores = pd.concat(list_boxscores)
    df_boxscores.rename(columns={'Unnamed: 0': 'player_game_index'}, inplace=True)

    return df_boxscores


def get_schedule_per_season(season_id: str):
    """
    :param season_id: str, season id of the season you want to query
    :return df_schedule: pd.DataFrame, contains schedule of a season
    """
    # get the season schedule
    call_season = leaguegamelog.LeagueGameLog(season=season_id)
    schedule = pd.concat(call_season.get_data_frames())

    return schedule


from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonplayerinfo


def get_clean_player_data(player_id: str):
    """

    :param player_id: str, APIs player_id
    :return df: pd.DataFrame, contains a players career stats
    """
    call_draft = commonplayerinfo.CommonPlayerInfo(player_id)
    tmp_draft = pd.concat(call_draft.get_data_frames())
    draft = tmp_draft.loc[~tmp_draft['DRAFT_YEAR'].isna()][
        ['PERSON_ID', 'DISPLAY_FIRST_LAST', 'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'GREATEST_75_FLAG', 'HEIGHT',
         'WEIGHT', 'LAST_AFFILIATION', 'POSITION']]

    # get a player
    call_player = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36="PerGame")
    df = pd.concat(call_player.get_data_frames())

    # get array of all team ids
    nba_teams = pd.DataFrame(teams.get_teams())
    team_ids = np.unique(nba_teams['id'])

    # columns to drop
    drop_list = ["Team_ID", "ORGANIZATION_ID", "SCHOOL_NAME", "RANK_PG_MIN", "RANK_PG_FGM", "RANK_PG_FGA",
                 "RANK_FG_PCT", "RANK_PG_FG3M", "RANK_PG_FG3A", "RANK_FG3_PCT", "RANK_PG_FTM", "RANK_PG_FTA",
                 "RANK_FT_PCT", "RANK_PG_OREB", "RANK_PG_DREB", "RANK_PG_REB", "RANK_PG_AST", "RANK_PG_STL",
                 "RANK_PG_BLK", "RANK_PG_TOV", "RANK_PG_PTS", "RANK_PG_EFF"]

    # ids of the nba teams - removes all-star appearances
    bool_nbateam = df['TEAM_ID'].isin(team_ids)

    # drop cols
    df = df.loc[:, ['PLAYER_ID', 'SEASON_ID', 'TEAM_ID', 'PLAYER_AGE', 'GP',
                    'GS', 'MIN', 'FG_PCT', 'FGA', 'FG3_PCT', 'FG3A', 'FT_PCT', 'FTA', 'OREB', 'DREB', 'REB',
                    'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]

    # remove all "other teams" e.g. all-star and totals
    df = df[bool_nbateam]

    # drop na's
    df = df.dropna()

    # keep first observations - what exactly are those?
    df = df[~df.index.duplicated(keep='first')]

    df2 = pd.merge(df, draft, left_on='PLAYER_ID', right_on='PERSON_ID')

    # convert height and weight

    feet = df2['HEIGHT'].apply(lambda x: int(x.split('-')[0]))
    inches = df2['HEIGHT'].apply(lambda x: int(x.split('-')[1]))

    tot_inches = feet * 12 + inches
    df2['HEIGHT_METER'] = np.round(tot_inches * 0.0254, 2)

    df2['WEIGHT_KG'] = np.round(df2['WEIGHT'].astype(int) * 0.453592, 2)

    return df2

def get_player_score(player_id: str):
    """

    :param player_id:
    :return:
    """
    data = pd.read_csv("./data/season_prediction/player_scores_16_20.csv")
    data_player = data[data['PLAYER_ID'] == player_id]
    return np.round(data_player['coef_perc_rank'].values[0], 2)

def get_season_data(player_id: str):

    data = pd.read_csv("./data/season_prediction/player_season_scores.csv")
    data_player = data[data['PLAYER_ID'] == player_id]
    tmp = data_player.sort_values(['SEASON_ID'])
    tmp['SEASON'] = data_player['SEASON_ID'].apply(lambda x: int(x.split('-')[0]))

    return tmp

def last_four_seasons(player_id, season_scores):
    """ Visualize players performance over last four seasons
    """

    tmp = season_scores[season_scores['PLAYER_ID'] == player_id]
    tmp = tmp.sort_values('SEASON_ID')
    tmp = tmp.reset_index()

    sns.lineplot(data=tmp, x="SEASON_ID", y="coef_perc_rank", ci=90)

    plt.title(tmp['DISPLAY_FIRST_LAST'].values[0])
    plt.show()

def get_player_salary(player_id: str):

    data = pd.read_csv("./data/data_assets/players_salaries_hist.csv")
    data = data.drop(['Unnamed: 0'], axis=1)
    tmp = pd.melt(data, id_vars=['id', 'player_names'])
    tmp['SEASON'] = tmp['variable'].apply(lambda x: int(x.split('/')[0]))
    tmp2 = tmp[tmp['id'] == player_id]

    print(tmp2)

    return tmp2


def get_player_image(player_id):
    """ Get a players image based on his id

    :param player_id: APIs player_id
    :return: Image, also opens the image
    """
    return Image.open(requests.get(f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", stream=True).raw)

def get_team_image(team_abb, season: str = '2021-21'):
    """

    :param team_abb:
    :return:
    """
    url_start = "http://stats.nba.com/media/img/teams/logos/season/"
    url_end = '_logo.svg'
    url = url_start + season + '/' + team_abb + url_end

    Image.open(requests.get(url, stream=True).raw)


def load_boxscore_schedule(data_path: str="./data/season_prediction/"):
    """

    :param data_path:
    :return:
    """

    # init storage
    list_schedule, list_boxscore, obs_schedule, obs_boxscore = list(), list(), list(), list()
    list_season_id = [['2014-15'], ['2015-16'], ['2016-17'], ['2017-18'],
                      ['2018-19'], ['2019-20'], ['2020-21']]

    # load and divide datafiles
    for file in sorted(listdir(data_path)):

        if file.startswith("boxscores"):
            tmp = pd.read_csv(data_path + file, dtype={'GAME_ID':str})
            list_boxscore.append(tmp)
            obs_boxscore.append(tmp.shape[0])

        elif file.startswith("schedule"):
            tmp = pd.read_csv(data_path + file, dtype={'GAME_ID':str})
            list_schedule.append(tmp)
            obs_schedule.append(tmp.shape[0])

        else:
            raise ValueError("Neither boxscore nor schedule!")

        print(file + ' loaded')

    # check check
    assert len(list_schedule) == len(list_boxscore), 'Not the same amount of seasons!'

    # transform to pandas dataframes
    return pd.concat(list_schedule), pd.concat(list_boxscore)