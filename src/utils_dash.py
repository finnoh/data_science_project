import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamdashboardbylastngames

def _player_selector():
    """ Helper for selecting a player with a selector

    :return tmp3: dict, label and value pairs for selector
    """
    # get_players returns a list of dictionaries, each representing a player.
    tmp = pd.DataFrame(players.get_active_players())  # just return the ids
    tmp2 = tmp[['full_name', 'id']]
    tmp3 = tmp2.rename(columns={"full_name": "label", "id": "value"})
    return tmp3


def _team_selector():
    """

    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'GAME_ID': str})
    tmp = df[['full_name', 'abbreviation']]
    tmp2 = tmp.rename(columns={"full_name": "label", "abbreviation": "value"})
    return tmp2


def _team_full_name(abbreviation):
    """

    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'GAME_ID': str})
    tmp = df[df['abbreviation'] == abbreviation]['full_name'].values

    return tmp[0].lower()


def _player_full_name(player_id):
    """

    :return:
    """
    df = pd.read_csv("players_data.csv", dtype={'id': str})
    player_name = df[df['id'] == player_id]['player_names'].values[0]
    player_pos = df[df['id'] == player_id]['position'].values[0]

    return player_name, player_pos


def _link_team_website(abbreviation):
    """

    :param abbreviation:
    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'TEAM_ID': str})
    tmp = df[df['abbreviation'] == abbreviation]['nickname'].values

    return tmp[0].lower()


def _get_team_id(abbreviation):
    """

    :param abbreviation:
    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'TEAM_ID': str})
    team_id = df[df['abbreviation'] == abbreviation]['TEAM_ID'].values

    return team_id[0]


def _get_mvp_id_team(team_id, season: str = '2020-21'):
    """

    :param abbreviation:
    :return:
    """

    df = pd.read_csv("playercareerstats_activeplayers.csv", dtype={'TEAM_ID': str, 'PLAYER_ID': str})

    subset = df[(df['TEAM_ID'] == team_id) & (df['SEASON_ID'] == season)]
    mvp = subset[subset['PTS'] == subset['PTS'].max()]

    mvp_id = mvp['PLAYER_ID'].values[0]
    mvp_age = int(mvp['PLAYER_AGE'].values[0])
    mvp_gp = mvp['GP'].values[0]
    mvp_pts, mvp_ast, mvp_reb = np.round(mvp['PTS'].values[0] / mvp_gp, 1), np.round(mvp['AST'].values[0] / mvp_gp, 1), np.round(mvp['REB'].values[0] / mvp_gp, 1)
    url_image = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(mvp_id)}.png"

    mvp_data = [mvp_id, mvp_age, mvp_pts, mvp_ast, mvp_reb, mvp_gp]

    return mvp_data, url_image


def _mvp_descr_builder(mvp_name, mvp_position, mvp_data):
    """

    :param mvp_name:
    :param mvp_position:
    :param mvp_data:
    :return:
    """

    rand = np.random.randint(low=0, high=6)
    rand_excellent = np.random.randint(low=0, high=6)
    rand_star = np.random.randint(low=0, high=6)

    word_excellent = ['phenomenal', 'excellent', 'stellar', 'amazing', 'outstanding']
    word_star = ['icon', 'all-star caliber', 'franchise-player', 'star-player', 'our mvp']

    if mvp_position == "G":
        position = "Guard"
    elif mvp_position == "F":
        position = "Forward"
    else:
        position = "Center"

    if rand == 0:
        sentence = f"Our {str(mvp_data[1])} year old {position} {mvp_name} scored {word_excellent[rand_excellent]} {str(mvp_data[2])} points last season."
    elif rand == 1:
        sentence = f"{mvp_name}'s ({mvp_position}) stats last season were {word_excellent[rand_excellent]} {str(mvp_data[2])} points while contributing an average {str(mvp_data[3])} assists per game."
    elif rand == 3:
        sentence = f"{mvp_data[5]} games last season make our {word_star[rand_star]} {position}, {mvp_name} one of our most busy players. At an age of {str(mvp_data[1])} he scored {str(mvp_data[2])} points while securing {str(mvp_data[4])} boards per game."
    elif rand == 4:
        sentence = f"{str(mvp_data[2])} points, {str(mvp_data[3])} assists and {str(mvp_data[4])} per game. Those are the stats of our  {str(mvp_data[1])} year old {word_star[rand_star]} {mvp_name}."
    else:
        sentence = f"At {str(mvp_data[1])} years of age, {mvp_name} leads the team with {str(mvp_data[2])} point per game. The {word_excellent[rand_excellent]} {position} also dishes out {str(mvp_data[3])} assists a night. He played {mvp_data[5]} games last season."

    return sentence

# def _team_lastngames(team_id, n, season: str = "2021-22"):
#     """
#
#     :param team_id:
#     :param n:
#     :param season:
#     :return:
#     """
#     call = teamdashboardbylastngames.TeamDashboardByLastNGames(team_id=str(team_id), last_n_games=n, season=season)
#     df = pd.concat(call.get_data_frames())
#     print(df)
#     return df