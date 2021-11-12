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
