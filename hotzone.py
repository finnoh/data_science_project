from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import matplotlib.pyplot as plt

def hotzone(player_id):
    team_player = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_normalized_dict()['CommonPlayerInfo'][0]
    shots_dict = shotchartdetail.ShotChartDetail(team_id=team_player['TEAM_ID'], player_id=player_id, context_measure_simple = 'FGA', season_type_all_star='Regular Season').get_normalized_dict()['Shot_Chart_Detail']
    shots_df = pd.DataFrame.from_dict(shots_dict)
    #print(shots_df.columns)
    #shots_df['SHOTS_MADE_FLAG'] = shots_df['SHOTS_MADE_FLAG']
    return shots_df #shots_df['LOC_X'], shots_df['LOC_Y'], shots_df['SHOT_MADE_FLAG']

if __name__ == "__main__":
    #x, y, made = hotzone('2544')
    #print(shots_df.head())
    #plt.scatter(x, y, c = made,  s = 0.5)
    #print(made[:5])
    shots = hotzone('2544')
    plt.scatter(shots['LOC_X'], shots['LOC_Y'], c = shots['SHOT_MADE_FLAG'], s = 0.5)
    #plt.legend()
    plt.title("Hot zone")
    plt.show()

# or use SHOT_TYPE + SHOT_DISTANCE to define hot zone for made points