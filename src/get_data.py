import pandas as pd
import numpy as np
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from PIL import Image
import requests
from tqdm import tqdm
import time


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

def get_clean_player_data(player_id: str):
    """

    :param player_id: str, APIs player_id
    :return df: pd.DataFrame, contains a players career stats
    """
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

    return df

def get_player_image(player_id):
    """ Get a players image based on his id

    :param player_id: APIs player_id
    :return: Image, also opens the image
    """
    return Image.open(requests.get(f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", stream=True).raw)