import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import copy
import matplotlib.ticker as mtick
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playercareerstats

pd.options.mode.chained_assignment = None

# define functions for data loading
def get_players_data():
    return pd.read_csv('players_data.csv').sort_values(by=['id']).reset_index(drop = True)

def get_players_salary():
    return pd.read_csv('players_salaries.csv').sort_values(by=['id'])

def get_players_stats():
    return pd.read_csv('playercareerstats.csv').sort_values(by=['PLAYER_ID'])

def get_physical_attributes():
    return pd.read_csv('data/rec_engine/further_attributes.csv')

def get_teams_data():
    return pd.read_csv('teams_data.csv')

def get_teams_salaries():
    teams_salaries = pd.read_csv('teams_salaries.csv')
    teams_salaries.loc[teams_salaries['Abb'] == 'UTH', 'Abb'] = 'UTA' # manual fix
    return teams_salaries

def get_boxscores(season):
    return pd.read_csv(f'data/season_prediction/boxscores_{season}.csv')

def get_playoffs(season):
    return pd.read_csv(f'data/season_prediction/playoffs_{season}.csv')

def get_player_scores():
    return pd.read_csv('data/season_prediction/player_season_scores.csv')

def get_2k_ratings():
    return pd.read_csv('data/rec_engine/nba2k_ratings_adj.csv')

#players_stats_agg = pd.read_csv('playercareerstats_agg.csv').sort_values(by=['PLAYER_ID']) # gewichtete Durchschnitte der letzten 3 Saisons: 1/3, 2/3, 3/3

players_data = get_players_data()
players_salaries = get_players_salary()
players_stats = get_players_stats()
players_physical = get_physical_attributes()
players_scores = get_player_scores()
players_nba2k = get_2k_ratings()
teams_data = get_teams_data()
teams_salaries = get_teams_salaries()

boxscores_20_21 = get_boxscores('20_21')
boxscores_19_20 = get_boxscores('19_20')
boxscores_18_19 = get_boxscores('18_19')
boxscores_17_18 = get_boxscores('17_18')
boxscores_16_17 = get_boxscores('16_17')
boxscores_15_16 = get_boxscores('15_16')
boxscores_14_15 = get_boxscores('14_15')


seasons = ['14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21']
boxscores_list = [boxscores_14_15, boxscores_15_16, boxscores_16_17, boxscores_17_18, boxscores_18_19, boxscores_19_20, boxscores_20_21] 


playoffs_players = pd.read_csv('data/rec_engine/playoffs_players.csv', dtype={'PLAYER_ID': str})
playoffs_boxscores = pd.read_csv('data/rec_engine/playoffs_boxscores.csv', dtype={'SEASON': str, 'PLAYER_ID': str})

# Define function to combine seasons based on specified weights
def combine_seasons(players_stats, player_id, weights, seasons):
    df = players_stats[players_stats['PLAYER_ID'] == player_id]
    
    season_0 = df[df['SEASON_ID'] == seasons[0]]

    if season_0.shape[0] == 0:
        season_0 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_0.shape[0] > 1:
        season_0 = season_0[season_0['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[0] # take the TOTAL performance (over all teams, the player played in during the season)
    else:
        season_0 = season_0.iloc[:,6:] * weights[0]

    season_1 = df[df['SEASON_ID'] == seasons[1]]
    if season_1.shape[0] == 0:
        season_1 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_1.shape[0] > 1:
        season_1 = season_1[season_1['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[1]
    else:
        season_1 = season_1.iloc[:,6:] * weights[1]
    
    
    season_2 = df[df['SEASON_ID'] == seasons[2]]
    if season_2.shape[0]  == 0:
        season_2 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_2.shape[0] > 1:
        season_2 = season_2[season_2['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[2]
    else:
        season_2 = season_2.iloc[:,6:] * weights[2]
        
    # combine weighted values
    values_pastSeasons = (season_0.values + season_1.values + season_2.values).flatten()
    
    if sum(values_pastSeasons) == 0:
        # can optionally print out the players name who has no data
        #player_name = list(players_data[players_data['id'] == player_id]['player_names'])[0]
        #print(f'No game data: {player_name} with id {player_id}')
        return 'NA'
    
    df_final = copy.deepcopy(df)
    df_final.iloc[-1, 6:] = values_pastSeasons

    df_final.iloc[-1, 1:3] = 'aggregated'
     
    dict_final = dict(df_final.iloc[0])
    return dict_final

# aggregate data of specified columns based on the three specified seasons, weights etc.
def aggregate_data(players_stats, seasons = ['2020-21', '2019-20', '2018-19'], w = [7/10, 2/10, 1/10], cols = None, rec_type = 'Similar', norm = True, current_season = True, output_table = False):
    # retrieve data of correct seasons
    players_stats = players_stats[(players_stats['SEASON_ID'] == seasons[0]) | 
                                  (players_stats['SEASON_ID'] == seasons[1]) | 
                                  (players_stats['SEASON_ID'] == seasons[2])].reset_index().drop(columns=['index'])

    # numeric columns which need to be adjusted for number of minutes played
    col_div = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    # define columns which need to be adapted
    if cols is not None:
        cols_adapted = [x for x in cols if x not in ['EXPERIENCE', 'HEIGHT', 'WEIGHT', 'Playmaking', 'Athleticism', 'Score']]
        players_stats = players_stats[cols_adapted]    
        col_idx =  [list(players_stats.columns).index(i) for i in col_div if i in cols_adapted]

    else:
        col_idx =  [list(players_stats.columns).index(i) for i in col_div]
        cols = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'EXPERIENCE', 'HEIGHT', 'WEIGHT', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'Playmaking', 'Athleticism', 'Score']

    # for output table: not divide by minutes played, otherwise yes
    for i in range(players_stats.shape[0]):
        n_min = players_stats["MIN"][i]
        for j in col_idx:
            if output_table == True:
                players_stats.iloc[i, j] /= n_min
                players_stats.iloc[i, j] *= 36
            else:
                players_stats.iloc[i, j] /= n_min

    # combine the three seasons to one row
    player_stats_agg_notTransformed = [combine_seasons(players_stats, player_id, w, seasons) for player_id in players_data['id']]
    try:
        ind_player_drop = player_stats_agg_notTransformed.index('NA')
    except ValueError:
        pass
    player_stats_agg_notTransformed = [x for x in player_stats_agg_notTransformed if x != 'NA']
    player_stats_agg_notTransformed = pd.DataFrame(player_stats_agg_notTransformed).sort_values(by=['PLAYER_ID']).reset_index(drop = True)

    # join further attributes
    if current_season:
        season_df = players_physical[players_physical['Season'] == 2020]
       
        player_scores = []
        player_playmaking = []
        player_athleticism = []
        for p_id in list(player_stats_agg_notTransformed['PLAYER_ID'].unique()):
            try:
                player_scores.append(list(players_scores[(players_scores['PLAYER_ID'] == p_id) & (players_scores['SEASON_ID'] == '2020-21')]['coef'])[0])
                p_name = list(players_data[players_data['id'] == p_id]['player_names'])[0]
                player_playmaking.append(list(players_nba2k[players_nba2k['player'] == p_name]['playmaking'])[0])
                player_athleticism.append(list(players_nba2k[players_nba2k['player'] == p_name]['athleticism'])[0])
            except:
                player_scores.append(0) # 0 as neutral element for the players not receiving a score
                p_name = list(players_data[players_data['id'] == p_id]['player_names'])[0]
                player_playmaking.append(list(players_nba2k[players_nba2k['player'] == p_name]['playmaking'])[0])
                player_athleticism.append(list(players_nba2k[players_nba2k['player'] == p_name]['athleticism'])[0])

        # add columns only for the 'Similar' recommendation option
        if rec_type == 'Similar':
            if 'Playmaking' in cols:
                player_stats_agg_notTransformed.insert(8, "Playmaking", player_playmaking)
            if 'Athleticism' in cols:
                player_stats_agg_notTransformed.insert(8, "Athleticism", player_athleticism)
            if 'Score' in cols:
                player_stats_agg_notTransformed.insert(8, "Score", player_scores)

        if 'EXPERIENCE' in cols:
            player_stats_agg_notTransformed.insert(8, "EXPERIENCE", [list(season_df[season_df['player_id'] == p_id]['Experience'])[0] for p_id in list(player_stats_agg_notTransformed['PLAYER_ID'].unique())])
        if 'HEIGHT' in cols:
            player_stats_agg_notTransformed.insert(8, "HEIGHT", [list(season_df[season_df['player_id'] == p_id]['Height (cm)'])[0] for p_id in list(player_stats_agg_notTransformed['PLAYER_ID'].unique())])
        if 'WEIGHT' in cols:
            player_stats_agg_notTransformed.insert(8, "WEIGHT", [list(season_df[season_df['player_id'] == p_id]['Weight (kg)'])[0] for p_id in list(player_stats_agg_notTransformed['PLAYER_ID'].unique())])
        

    players_stats_agg = copy.deepcopy(player_stats_agg_notTransformed)

    # normalize the data to have zero mean and unit variance
    if norm == True:
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(players_stats_agg.iloc[:,5:])
        players_stats_agg.iloc[:,5:] = norm_data

    # drop not interesting columns
    players_stats_agg = players_stats_agg.drop(columns=['GP', 'GS', 'MIN'])
    player_stats_agg_notTransformed = player_stats_agg_notTransformed.drop(columns=['GP', 'GS', 'MIN'])

    return players_stats_agg, player_stats_agg_notTransformed


## Define help functions
def get_playerID(name):
    try:
        return list(players_data[players_data['player_names'] == name]['id'])[0]
    except IndexError:
        print('Please enter a valid position.')
        pass

def adj_position(pos):
    if pos[0] in ['C', 'F', 'G']:
        return pos[0]
    else:
        print('Please enter a valid position.')
        pass

def visualize_capspace(input_data, labels, team):
    x_values = ['2021/22', '2022/23', '2023/24', '2024/25']
    colors = ['blue', 'green', 'red']
    y_values = [list(input_data.iloc[i, 3:]) for i in range(input_data.shape[0])]
            
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(len(y_values)):
        ax.plot(x_values, y_values[i], label = labels[i], color= colors [i])
    ax.set(title = f' Cap Space Development of {team}',
           xlabel = "Season",
           ylabel = "Cap Space in $")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.0f $'))
    plt.legend()

    return plt.show()


def visualize_capspace_team(team_abb):
    if team_abb in list(teams_salaries['Abb']):
        capspace_team = teams_salaries[teams_salaries['Abb'] == team_abb].reset_index(drop = True)
        y_values = capspace_team.iloc[0, 3:]
    else:
        print('Please input a correct abbreviation of an NBA team')
        return 0
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(['2021/22', '2022/23', '2023/24', '2024/25'], list(y_values))
    ax.set(title = f' Cap Space Development of {team_abb}',
           xlabel = "Season",
           ylabel = "Cap Space (in")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.0f $'))
    plt.show()
    return capspace_team

def visualize_capspace_team_plotly(team_abb):
    if team_abb in list(teams_salaries['Abb']):
        capspace_team = teams_salaries[teams_salaries['Abb'] == team_abb].reset_index(drop=True)
        y_values = capspace_team.iloc[0, 3:]
    else:
        print('Please input a correct abbreviation of an NBA team')
        return 0

    df_plot = pd.DataFrame(data={'Season': ['2021/22', '2022/23', '2023/24', '2024/25'], 'Cap Space': list(y_values)})

    fig = px.line(df_plot, x="Season", y="Cap Space", title=f' Cap Space Development')

    return fig.update_layout(template="simple_white")

def luxury_tax(cap_space):
    cap_taxed = -(cap_space + (136606000 - 112414000)) # difference between Cap Maximum ($112,414,000) & Luxury Tax Threshold ($136,606,000)
    tax = 0

    max_perc = (3.75 + 0.50*(((cap_taxed)//5000000) -4)) # maximum percentage tax based on 5M increments
    tax_perc = np.append(np.array([1.50, 1.75, 2.50, 3.25]), np.arange(3.75, max_perc + 0.5, 0.5))
        
    # iterate through the taxed cap and add the luxury tax
    if cap_taxed < 0:
        return tax
    else:
        i = 0
        while (cap_taxed != 0):
            if (cap_taxed - 5000000) > 0:
                cap_taxed -= 5000000
                tax += 5000000*tax_perc[i]
                i += 1
                
            else:
                tax += cap_taxed*tax_perc[i]
                break
                
        return tax/10

# function to determine the starting five of a team based on a boxscore
def starting_five(boxscores = boxscores_20_21, team_abb = 'LAL', names = False, current_season = True):
    positions = {'F': 2, 'C': 1, 'G': 2} # compisition of positions to retrieve starting five
    data_team = boxscores[(boxscores['TEAM_ABBREVIATION'] == team_abb) & (boxscores['START_POSITION'].notnull())].loc[:, ['PLAYER_ID', 'START_POSITION']]
    players_team = list(players_data[players_data['team'] == team_abb]['id'])

    if list(data_team['START_POSITION'].unique()) != list(positions.keys()):
        print('Error')

    # find player for each position based on how often he started in this position
    players_pos = list()
    for pos in positions.keys():
        data_team_pos = data_team[data_team['START_POSITION'] == pos]['PLAYER_ID'].astype(int)
        count_pos = Counter(data_team_pos)
        count_pos = dict(sorted(count_pos.items(), key=lambda item: item[1], reverse=True))
        del_players = []
        if current_season:
            for i in range(len(count_pos)):
                player = list(count_pos.keys())[i]
                if player not in players_team: # only keep players which are still active and belong to team at end of last season
                    del_players.append(player)

            for p in del_players:
                del count_pos[p]
        players_pos.append(count_pos)


    # delete player from positions where he played less frequently
    players = [list(players_pos[i].keys()) for i in range(len(positions))]
    players = dict(Counter([x for l in players for x in l]))
    dupl_players = [k for k,v in players.items() if v > 1]
    for dupl_pl in dupl_players:
        counts = []
        for i in range(len(players_pos)):
            try:
                counts.append(players_pos[i][dupl_pl])

            except KeyError:
                counts.append(0)
        keep_pos = np.argmax(counts)
        for j in range(len(players_pos)):
            if j == keep_pos:
                continue
            try:
                del players_pos[j][dupl_pl]

            except KeyError:
                continue

    # determine final starting five
    start_five = {}
    for i in range(len(positions)):
        pos = list(positions.keys())[i]
        dict_pos = players_pos[i]
        pos_players = list(dict_pos.keys())[:(positions[pos])]
        for i in range(len(pos_players)):
            if names:
                try:
                    name = list(players_data[players_data['id'] == pos_players[i]]['player_names'])[0]
                except IndexError:
                    name = commonplayerinfo.CommonPlayerInfo(pos_players[i]).get_data_frames()[0]['DISPLAY_FIRST_LAST'][0]
                start_five[name] = pos 

            else:
                start_five[pos_players[i]] = pos                               

    return start_five


# Dimensionality reduction: performed on given aggregated data
def embeddings(option: str, stats_agg, stats_agg_notTransformed, dim = 2):
    data_names = list(players_data['player_names'])
    players_stats = copy.deepcopy(stats_agg.iloc[:,:5])

    if option == "spectral":
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components = dim, random_state = 42, n_neighbors = stats_agg.shape[0]//75)

    elif option == 'tsne':
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components = dim)

    elif option == 'umap':
        import umap.umap_ as umap
        embedding = umap.UMAP(n_components = dim, random_state = 42)

    elif option == 'pca':
        from sklearn.decomposition import PCA
        embedding = PCA(n_components = dim)

    else:
        print('Please enter a valid embedding.')

    stats_transformed = embedding.fit_transform(stats_agg.iloc[:,5:])

    players_stats["embedding_1"] = stats_transformed[:,0]
    players_stats["embedding_2"] = stats_transformed[:,1]

    if dim == 3:
        players_stats["embedding_3"] = stats_transformed[:,2]

    
    return players_stats, embedding, players_data['position'], data_names, stats_agg_notTransformed.iloc[:,5:]


#  Class definition of the Engine
class RecommendationEngine:
    def __init__(self, data, replacing_player, option, distance_measure = 'L2', w = [7/10, 2/10, 1/10], cols_sel = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']):
        self.stats = data
        self.option = option
        self.player_name = replacing_player
        self.player_id = players_data[players_data["player_names"] == replacing_player]['id'].iloc[0]
        self.position = adj_position(commonplayerinfo.CommonPlayerInfo(self.player_id).get_data_frames()[0]['POSITION'][0])
        self.team = self.team_lastSeason()
        self.distance_measure = distance_measure
        self.w = w
        self.cols_sel = cols_sel
            
    def recommend(self):   

        # throw error if one of those attributes is used in conjunction with the wrong recommendation option
        if (self.option == 'Fit') & len(set(['Playmaking', 'Athleticism', 'Score']).intersection(set(self.cols_sel))) > 0:
            raise ValueError("The 'Fit' option cannot be just in conjunction with one of the following attributes: 'Playmaking', 'Athleticism', 'Score'.\nPlease change your selected features.")

        # adjust data
        stats_repl_player = self.stats[self.stats['PLAYER_ID'] == self.player_id].iloc[:,5:].to_numpy() # get data from player to be replaced
        stats = self.stats[players_data['team'] != self.team] # exclude players from same team
        stats_num = stats.iloc[:,5:].to_numpy()        
       
        if stats_repl_player.shape[0] != 0:
    
            if self.option == 'Similar':
                # compute closest distances in high-dimensional space
                closest_idx, closest_distances, closest_distances_details = self.distance_comp(stats_repl_player, stats_num, self.distance_measure)
                
            elif self.option == 'Fit':               
                # get starting five of team (without player to be replaced)
                start_five_team = list(starting_five(boxscores_20_21, self.team, names = False).keys())
                start_five_team.remove(self.player_id)
                
                # get aggregate statistics of the team of the player to be replaced
                data_team = pd.concat([self.stats[self.stats['PLAYER_ID'] == start_five_team[i]] for i in range(len(start_five_team))])
                data_team = np.abs(np.array(data_team.iloc[:,5:].sum(axis=0)))

                # get desired attributes for team via the role model players (incl. clustering)
                ideal_player = self.model_teams(stats_repl_player, data_team)
                
                # get closest players and remove double indexing
                closest_idx, closest_distances, closest_distances_details = self.distance_comp(ideal_player, stats_num, self.distance_measure)
            
            team_salary = self.team_salary()
            salary_input_player = self.player_salary(self.player_name)


            # create list of best recommendations
            closest_players = [{'player': self.player_name,
                                'distance': 0,
                                'distance_details': stats_repl_player[0], 
                                'luxury_tax': 0}]
            
            # determine salary changes for each player
            for i in range(len(closest_idx)):
                id_player = stats.reset_index()['PLAYER_ID'][closest_idx[i]]
                name_player = players_data[players_data['id'] == id_player]['player_names'].iloc[0]
                salary_rec_player = self.player_salary(name_player)
                change_salary = self.change_salary(list(salary_input_player.iloc[0,1:]), list(salary_rec_player.iloc[0,1:]))
                new_team_salary = self.new_team_salary(change_salary, team_salary)
                
                #print('Old', team_salary)
                #print('New', new_team_salary)

                luxury_tax_player = [(luxury_tax(new_team_salary.iloc[0, i]) - luxury_tax(team_salary.iloc[0, i])) for i in range(3, team_salary.shape[1])]

                #print(luxury_tax_player)

                stats_player = stats_num[closest_idx[i], :]
                closest_players.append({'player': name_player,
                                        'distance': closest_distances[i],
                                        'distance_details': stats_player,
                                        'luxury_tax': luxury_tax_player})
            
            # add distance details to output dataframe
            result_df = pd.DataFrame(closest_players)
            cols_adding_df = list(stats.columns[5:])
            for i, col in enumerate(cols_adding_df):
                res_cols = [round(result_df['distance_details'].iloc[player][i], 2) for player in range(result_df.shape[0])]
                result_df[col] = res_cols
            result_df['luxury_tax'] = result_df['luxury_tax'].apply(np.sum)
            result_df.drop(columns = ['distance_details'], inplace = True)

            #print(result_df)

            rec_player = closest_players[1]['player']
               
            
            #print(f"Input Player: {self.player_name} (Team: {self.team})")
            #print('Salary:')
            salary_input_player = self.player_salary(self.player_name)
            #print(salary_input_player)

            
            #print(f'\nRecommended Player: {rec_player}')
            #print('Salary:')
            salary_rec_player = self.player_salary(rec_player)
            #print(salary_rec_player)
            
            #print('-> Change in salary:')
            change_salary = self.change_salary(list(salary_input_player.iloc[0,1:]), list(salary_rec_player.iloc[0,1:]))
            #display(change_salary)
            
            #print('Salary Input Team:')
            team_salary = self.team_salary()
            #display(team_salary)
            
        
            #print('New Salary Input Team:')
            new_team_salary = self.new_team_salary(change_salary, team_salary)
            #display(new_team_salary)

            # Take with caution because also many players still have 0 salary
            #print(f"Change in projected luxury tax: {[(luxury_tax(new_team_salary.iloc[0, i]) - luxury_tax(team_salary.iloc[0, i])) for i in range(3, team_salary.shape[1])]}")
            
            return rec_player, result_df
        
        print("No data available for this player in the last season")
        pass
    
    # compute distances of measure
    def distance_comp(self, node, nodes, distance_measure, topN = 5):
        node, nodes = np.asarray(node), np.asarray(nodes)
        if distance_measure == 'L2':
            distances_detailed = (nodes - node)**2
            distances = np.sum((nodes - node)**2, axis=1)
        elif distance_measure == 'L1':
            distances_detailed = np.abs(nodes - node)
            distances = np.sum(np.abs(nodes - node), axis=1)
        else:
            print('Please enter a valid distance measure.')
            pass

        topN_ids = np.argsort(distances)[: topN]
        return topN_ids[:topN + 1], np.sort(distances)[:topN], distances_detailed[topN_ids, :]
    
    # retrieve salary of player
    def player_salary(self, rec_player):
        return players_salaries[players_salaries['player_names'] == rec_player]
    
    # compute change in salary
    def change_salary(self, df_inputplayer, df_recplayer):
        # input - recommended
        change = [float(df_inputplayer[i]) - float(df_recplayer[i]) for i in range(1, len(df_inputplayer))] 
        return change
    
    # retrieve salary of team
    def team_salary(self):
        abb_team = list(players_data[players_data['id'] == self.player_id]['team'])[0]
        return teams_salaries[teams_salaries['Abb'] == abb_team]
    
    # compute new team salary
    def new_team_salary(self, change_salary, df_old_salary):
        df_new_salary = copy.deepcopy(df_old_salary)
        for i in range(len(change_salary)):
            df_new_salary.iloc[0, 3+i] += change_salary[i]
        return df_new_salary
    
    # get team of last players
    def team_lastSeason(self):
        return list(players_data[players_data['id'] == self.player_id]['team'])[0]
    
    # compute limit salary of team
    def limit_salary_team(self, team_salary):
        df_limit_salary = copy.copy(team_salary)
        for i in range(3, df_limit_salary.shape[1]):
            if df_limit_salary.iloc[0, i] > 0:
                pass
            else:
                df_limit_salary.iloc[0, i] *= 1.1 # may overdraw another 10 %
        return df_limit_salary
    
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # get role model values from playoff teams
    def model_teams(self, input_player, agg_data_input_team):
        teams = {'20-21':['MIL', 'PHX', 'LAC', 'ATL'], '19-20':['LAL', 'MIA', 'DEN', 'BOS'], '18-19':['TOR', 'GSW', 'MIL', 'POR'], '17-18': ['GSW', 'CLE', 'HOU', 'BOS']} 
        team_ind = 0
        start_players_replaced_stats = []

        num_teams = sum([len(teams[x]) for x in teams if isinstance(teams[x], list)]) # count total number of model teams
        stats_teams = np.zeros((num_teams, len(self.cols_sel) - 8)) # not store values for player_id : team_abbreviation + GS, GP, MIN
        for i, season in enumerate(teams):  # iterate through all seasons
            seasons_past = list(reversed(seasons[seasons.index(season) - 3 : seasons.index(season)]))
            agg_data_seasons, _ = aggregate_data(players_stats, [f"20{season}" for season in seasons_past], self.w, self.cols_sel)
            season_features = f"20{seasons_past[0][:2]}"
            season_df = players_physical[players_physical['Season'] == int(season_features)]

            scaler = StandardScaler()
            norm_data = scaler.fit_transform(season_df.iloc[:,1:4])
            season_df.iloc[:,1:4] = norm_data
            
            for j in range(len(teams[season])): # iterate through all teams
                team = teams[season][j]
                s_five = playoff_player(season, team) # retrieve starting five
                stats_players = []
                dist_to_input_player = []
                players_not_found = []
                for s_five_player in s_five.keys(): # iterate through all players of starting five and compute distance
                    stats_player = agg_data_seasons[agg_data_seasons['PLAYER_ID'] == int(s_five_player)].iloc[:,5:] # with int() conversion: not good style
                    
                    # add additional attributes
                    if 'WEIGHT' in self.cols_sel:
                        stats_player['WEIGHT'] = list(season_df[season_df['player_id'] == int(s_five_player)]['Weight (kg)'])[0]
                    if 'HEIGHT' in self.cols_sel:
                        stats_player['HEIGHT'] = list(season_df[season_df['player_id'] == int(s_five_player)]['Height (cm)'])[0]
                    if 'EXPERIENCE' in self.cols_sel:
                        stats_player['EXPERIENCE'] = list(season_df[season_df['player_id'] == int(s_five_player)]['Experience'])[0]

                    stats_players.append(stats_player)
                    _, distance, _ = self.distance_comp(input_player, stats_player.to_numpy(), self.distance_measure, topN = 2)
                    try:
                        dist_to_input_player.append(distance[0])
                    except IndexError: # e.g. rookies, players retired many years ago
                        players_not_found.append(s_five_player)
                for player in players_not_found:
                    ind_player = list(s_five.keys()).index(player)
                    avg_player = np.mean(pd.concat(stats_players).to_numpy(), axis = 0)
                    _, distance, _ = self.distance_comp(input_player, avg_player, self.distance_measure, topN = 2)
                    dist_to_input_player.insert(ind_player, distance[0])

                # remove closest player & save as possible target player (via position)
                ind_same_position = [i for i in range(len(s_five)) if list(s_five.values())[i] == self.position]
                dist_relevant = [dist_to_input_player[i] for i in ind_same_position]
                start_player_replaced_pos = np.argmin(dist_relevant) # relevant index inside position group
                start_player_replaced = ind_same_position[start_player_replaced_pos] # relevant index inside entire starting five
                start_player_replaced_stats = stats_players.pop(start_player_replaced)
                start_players_replaced_stats.append(start_player_replaced_stats)

                # aggregate performance of remaining players of team
                data_team = np.abs(np.array(pd.concat(stats_players).sum(axis=0)))
                stats_teams[team_ind,:] = data_team
                team_ind += 1
                
        
        # cluster the performance of the teams based on the remaining 4 players
        range_n_clusters = np.arange(2, stats_teams.shape[0]) # clusters from 2-15
        silhouette_avg = []
        sum_of_squared_distances = []
        for num_clusters in range_n_clusters:
        
            # initialise kmeans
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(stats_teams)
            cluster_labels = kmeans.labels_
            
            # silhouette score
            silhouette_avg.append(silhouette_score(stats_teams, cluster_labels))                   
            sum_of_squared_distances.append(kmeans.inertia_)

        k_opt = range_n_clusters[np.argmax(silhouette_avg)] 
        kmeans = KMeans(n_clusters = k_opt).fit(stats_teams)
        model_teams_labels = kmeans.labels_
        cluster_pred = kmeans.predict(data_team.reshape(1, -1))[0] # predict cluster of input team
        
        ind_role_teams = [i for i, label in enumerate(model_teams_labels) if label == cluster_pred]
        
        repl_player_role_teams = [start_players_replaced_stats[i] for i in ind_role_teams]
        repl_player_agg = pd.concat(repl_player_role_teams).mean(axis=0) # compute ideal player by averaging over the missing players

        return repl_player_agg

# get starting five from playoff data
def playoff_player(season, team_abb, names = False):
    season_mapping = {'20-21': '2020', '19-20': '2019', '18-19': '2018', '17-18': '2017'}
    season = season_mapping[season]
    positions = {'F': 2, 'C': 1, 'G': 2}
    data_team = playoffs_boxscores[playoffs_boxscores['SEASON'] == season]
    data_team = data_team[data_team['TEAM_ABBREVIATION'] == team_abb].loc[:, ['PLAYER_ID', 'MIN']]
    potential_players = list(data_team['PLAYER_ID'].dropna().unique()) # retrieve all players who played

    # get positions of players
    positions_players = [list(playoffs_players[playoffs_players['PLAYER_ID'] == p_id]['POS'])[0] for p_id in potential_players]

    players_sec = list()

    for player in potential_players:
        data_player = data_team[data_team['PLAYER_ID'] == player]
        sec_played = data_player['MIN'].sum()
        players_sec.append(sec_played) # save the number of seconds played

    # retrieve starting five
    start_five = {}
    for i in range(len(positions)):
        pos = list(positions.keys())[i]
        n_player = positions[pos]
        ind_players = [i for i, e in enumerate(positions_players) if e == pos]
        sec_players = [players_sec[i] for i in ind_players]
        ind_most_played = list(np.argsort(sec_players))[-n_player:] # get n players which played the most in this position

        for i in ind_most_played:
            player_id = potential_players[ind_players[i]]
            if names:
                name = list(playoffs_players[playoffs_players['PLAYER_ID'] == player_id]['NAME'])[0]
                start_five[name] = pos 
            else:
                start_five[player_id] = pos                               
    return start_five


# Exemplary execution of recommmendation
if __name__ == "__main__":
    w = [7/10, 2/10, 1/10]
    rec_type = 'Similar'
    cols_sel = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'EXPERIENCE', 'HEIGHT', 'WEIGHT', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
    #cols_sel += ['Playmaking', 'Athleticism', 'Score']
    stats_agg, stats_agg_notTransformed = aggregate_data(players_stats, ['2020-21', '2019-20', '2018-19'], w, cols_sel, rec_type)
    #data_emb, emb, _, _, _ = embeddings('spectral', stats_agg, stats_agg_notTransformed, dim=3)
    sample_recommendation = RecommendationEngine(stats_agg, "LeBron James", rec_type, 'L2', w, cols_sel).recommend()
