import pandas as pd
from nba_api.stats.static import players

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

def _link_team_website(abbreviation):
    """

    :param abbreviation:
    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'GAME_ID': str})
    tmp = df[df['abbreviation'] == abbreviation]['nickname'].values

    return tmp[0].lower()
