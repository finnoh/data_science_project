import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors
import requests
from bs4 import BeautifulSoup
import copy
import matplotlib.ticker as mtick
import time
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playercareerstats


def get_players_data():
    return pd.read_csv('players_data.csv').sort_values(by=['id']).reset_index(drop = True)

def get_players_salary():
    return pd.read_csv('players_salaries.csv').sort_values(by=['id'])

def get_players_stats():
    return pd.read_csv('playercareerstats.csv').sort_values(by=['PLAYER_ID'])

def get_teams_data():
    return pd.read_csv('teams_data.csv')

def get_teams_salaries():
    teams_salaries = pd.read_csv('teams_salaries.csv')
    teams_salaries.loc[teams_salaries['Abb'] == 'UTH', 'Abb'] = 'UTA' # manual fix -> correct in other notebook producing the data
    return teams_salaries

def get_boxscores(season):
    return pd.read_csv(f'data/season_prediction/boxscores_{season}.csv')

def get_playoffs(season):
    return pd.read_csv(f'data/season_prediction/playoffs_{season}.csv')

#players_stats_agg = pd.read_csv('playercareerstats_agg.csv').sort_values(by=['PLAYER_ID']) # gewichtete Durchschnitte der letzten 3 Saisons: 1/3, 2/3, 3/3

players_data = get_players_data()
players_salaries = get_players_salary()
players_stats = get_players_stats()
teams_data = get_teams_data()
teams_salaries = get_teams_salaries()

boxscores_20_21 = get_boxscores('20_21')
boxscores_19_20 = get_boxscores('19_20')
boxscores_18_19 = get_boxscores('18_19')
boxscores_17_18 = get_boxscores('17_18')
boxscores_16_17 = get_boxscores('16_17')
boxscores_15_16 = get_boxscores('15_16')
boxscores_14_15 = get_boxscores('14_15')

#playoffs_20_21 = get_playoffs('20_21')
#playoffs_19_20 = get_playoffs('19_20')
#playoffs_18_19 = get_playoffs('18_19')
#playoffs_17_18 = get_playoffs('17_18')

playoffs_players = pd.read_csv('data/rec_engine/playoffs_players.csv', dtype={'PLAYER_ID': str})
playoffs_boxscores = pd.read_csv('data/rec_engine/playoffs_boxscores.csv', dtype={'SEASON': str, 'PLAYER_ID': str}) # adjust in model_teams.py


## Optional: change weights for aggregating seasonal data

def combine_seasons(players_stats, player_id, weights, seasons):
    df = players_stats[players_stats['PLAYER_ID'] == player_id]
    
    season_0 = df[df['SEASON_ID'] == seasons[0]]
    if season_0.shape[0] == 0:
        season_0 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_0.shape[0] > 1:
        season_0 = season_0[season_0['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[0]
    else:
        season_0 = season_0.iloc[:,6:] * weights[0] #* 1/2
  
    season_1 = df[df['SEASON_ID'] == seasons[1]]
    if season_1.shape[0] == 0:
        season_1 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_1.shape[0] > 1:
        season_1 = season_1[season_1['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[1]
    else:
        season_1 = season_1.iloc[:,6:] * weights[1] #* 2/6
    
    
    season_2 = df[df['SEASON_ID'] == seasons[2]]
    if season_2.shape[0]  == 0:
        season_2 = pd.DataFrame(np.zeros((1, len(df.columns) -6)))
    elif season_2.shape[0] > 1:
        season_2 = season_2[season_2['TEAM_ABBREVIATION'] == 'TOT'].iloc[:,6:] * weights[2]
    else:
        season_2 = season_2.iloc[:,6:] * weights[2] #* 1/6
        
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

def aggregate_data(players_stats, seasons = ['2020-21', '2019-20', '2018-19'], w = [7/10, 2/10, 1/10], cols = None, norm = True, current_season = True):
    players_stats = players_stats[(players_stats['SEASON_ID'] == seasons[0]) | 
                                  (players_stats['SEASON_ID'] == seasons[1]) | 
                                  (players_stats['SEASON_ID'] == seasons[2])].reset_index().drop(columns=['index'])

    #cols = [list(players_stats.columns)[i] for i in cols]

    col_div = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    if cols is not None:
        players_stats = players_stats[cols]    
        col_idx =  [list(players_stats.columns).index(i) for i in col_div if i in cols]

    else:
        col_idx =  [list(players_stats.columns).index(i) for i in col_div]

    for i in range(players_stats.shape[0]):
        n_min = players_stats["MIN"][i] #  per 'GP' or per 'MIN'?
        for j in col_idx:
            players_stats.iloc[i, j] /= n_min

    player_stats_agg_notTransformed = [combine_seasons(players_stats, player_id, w, seasons) for player_id in players_data['id']]
    try:
        ind_player_drop = player_stats_agg_notTransformed.index('NA')
    except ValueError:
        pass
    player_stats_agg_notTransformed = [x for x in player_stats_agg_notTransformed if x != 'NA']
    player_stats_agg_notTransformed = pd.DataFrame(player_stats_agg_notTransformed).sort_values(by=['PLAYER_ID']).reset_index(drop = True)
    
    players_stats_agg = copy.deepcopy(player_stats_agg_notTransformed)

    if norm == True:
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(players_stats_agg.iloc[:,5:])
        players_stats_agg.iloc[:,5:] = norm_data

    players_stats_agg = players_stats_agg.drop(columns=['GP', 'GS', 'MIN'])
    player_stats_agg_notTransformed = player_stats_agg_notTransformed.drop(columns=['GP', 'GS', 'MIN'])
    return players_stats_agg, player_stats_agg_notTransformed


#players_stats_agg, player_stats_agg_notTransformed = aggregate_data(players_stats, [7/10, 2/10, 1/10])


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
           ylabel = "Cap Space in $") # can be improved
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.0f $')) # can be improved
    #plt.fill_between(x_values,y_values[0], y_values[-1],color="None",hatch=".",edgecolor="r")
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
    ax.plot(['2021/22', '2022/23', '2023/24', '2024/25'], list(y_values)) # 2020/21
    ax.set(title = f' Cap Space Development of {team_abb}',
           xlabel = "Season",
           ylabel = "Cap Space (in")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.0f $')) # can be improved
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

    max_perc = (3.75 + 0.50*(((cap_taxed)//5000000) -4))
    tax_perc = np.append(np.array([1.50, 1.75, 2.50, 3.25]), np.arange(3.75, max_perc + 0.5, 0.5))
        
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
                
        return tax

def starting_five(boxscores = boxscores_20_21, team_abb = 'LAL', names = False, current_season = True):
    positions = {'F': 2, 'C': 1, 'G': 2}
    data_team = boxscores[(boxscores['TEAM_ABBREVIATION'] == team_abb) & (boxscores['START_POSITION'].notnull())].loc[:, ['PLAYER_ID', 'START_POSITION']]
    players_team = list(players_data[players_data['team'] == team_abb]['id'])

    if list(data_team['START_POSITION'].unique()) != list(positions.keys()):
        print('Error')

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
                    name = commonplayerinfo.CommonPlayerInfo(pos_players[i]).get_data_frames()[0]['DISPLAY_FIRST_LAST'][0] #pos_players[i]
                start_five[name] = pos 

            else:
                start_five[pos_players[i]] = pos                               

    return start_five


## Dimensionality reduction

def embeddings(option: str, stats_agg, stats_agg_notTransformed, dim = 2):
    data_names = list(players_data['player_names'])
    players_stats = copy.deepcopy(stats_agg.iloc[:,:5])

    if option == "spectral":
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components = dim, random_state = 42, n_neighbors = stats_agg.shape[0]//75)
        #stats_transformed = embedding.fit_transform(players_stats_agg.iloc[:,5:])

    elif option == 'tsne':
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components = dim)
        #stats_transformed = embedding.fit_transform(players_stats_agg.iloc[:,5:])

    elif option == 'umap':
        import umap.umap_ as umap
        embedding = umap.UMAP(n_components = dim, random_state = 42)
        #stats_transformed = embedding.fit_transform(players_stats_agg.iloc[:,5:])

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

    fig, ax = plt.subplots()
    sns.scatterplot(players_stats_spectral["embedding_1"], players_stats_spectral["embedding_2"], hue=players_data['position'], legend='full')
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(data_names[sel.index]))
    plt.show()

# pro position:
# pos = 'F'
# data_pos = players_stats_agg[players_data['position'] == pos].iloc[:,5:]



#  Class definition

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
        #ids_samePosition = list(players_data[players_data["position"] == self.position]['id'])
        #stats = players_stats_agg[players_stats_agg['PLAYER_ID'].isin(ids_samePosition)] # get only players of same position
        
        stats_repl_player = self.stats[self.stats['PLAYER_ID'] == self.player_id].iloc[:,5:].to_numpy() # get data from player to be replaced
        stats = self.stats[players_data['team'] != self.team] # exclude players from same team
        stats_num = stats.iloc[:,5:].to_numpy()        
        if stats_repl_player.shape[0] != 0:
    
            #model = NearestNeighbors(n_jobs = -1).fit(stats_num) # fit nearest neighbor model to all remaining players

            if self.option == 'Similar':
                #closest_distances, closest_idx = model.kneighbors(stats_repl_player.to_numpy(), n_neighbors = 5, return_distance = True) # get closest players
                #closest_distances, closest_idx = closest_distances[0], closest_idx[0] # remove double indexing
                closest_idx, closest_distances, closest_distances_details = self.distance_comp(stats_repl_player, stats_num, self.distance_measure)
                
            elif self.option == 'Fit':               
                # get starting five of team (without player to be replaced)
                start_five_team = list(starting_five(boxscores_20_21, self.team, names = False).keys())
                start_five_team.remove(self.player_id)
                
                # get aggregate statistics of the team of the player to be replaced
                data_team = pd.concat([self.stats[self.stats['PLAYER_ID'] == start_five_team[i]] for i in range(len(start_five_team))])
                data_team = np.abs(np.array(data_team.iloc[:,5:].sum(axis=0)))

                # get desired attributes for team -> to be adjusted via Clustering!
                ideal_player = self.model_teams(stats_repl_player, data_team)
                
                '''
                maxs = self.get_maxs_teams()
                # compute ideal complementary player for team
                diff = np.abs(maxs - data_team)
                dist_attributes = self.softmax(diff)
                #ideal_player = dist_attributes * maxs * maxs.sum() # pro Attribut um x% verbessern (1+) # data_team
                ideal_player = dist_attributes * maxs # * np.linalg.norm(maxs, ord = 2)
                ideal_player = maxs - data_team     
                print(ideal_player.shape)
                '''
                
                # get closest players and remove double indexing
                #closest_distances, closest_idx = model.kneighbors(ideal_player.reshape(1, -1), n_neighbors = 5, return_distance = True)
                #closest_distances, closest_idx = closest_distances[0], closest_idx[0]
                closest_idx, closest_distances, closest_distances_details = self.distance_comp(ideal_player, stats_num, self.distance_measure)
            
            team_salary = self.team_salary()
            salary_input_player = self.player_salary(self.player_name)
            #print(stats_repl_player[0])


            # create list of best recommendations
            closest_players = [{'player': self.player_name,
                                'distance': 0,
                                'distance_details': stats_repl_player[0], 
                                'luxury_tax': 0}]
            for i in range(len(closest_idx)):
                id_player = stats.reset_index()['PLAYER_ID'][closest_idx[i]]
                name_player = players_data[players_data['id'] == id_player]['player_names'].iloc[0]
                salary_rec_player = self.player_salary(name_player)
                change_salary = self.change_salary(list(salary_input_player.iloc[0,1:]), list(salary_rec_player.iloc[0,1:]))
                new_team_salary = self.new_team_salary(change_salary, team_salary)
                luxury_tax_player = [(luxury_tax(new_team_salary.iloc[0, i]) - luxury_tax(team_salary.iloc[0, i])) for i in range(3, team_salary.shape[1])]

                stats_player = stats_num[closest_idx[i], :]
                closest_players.append({'player': name_player,
                                        'distance': closest_distances[i],
                                        'distance_details': stats_player, #closest_distances_details[i],
                                        'luxury_tax': luxury_tax_player})
            

            result_df = pd.DataFrame(closest_players)
            cols_adding_df = list(stats.columns[5:])
            for i, col in enumerate(cols_adding_df):
                res_cols = [round(result_df['distance_details'].iloc[player][i], 2) for player in range(result_df.shape[0])]
                result_df[col] = res_cols
            result_df['luxury_tax'] = result_df['luxury_tax'].apply(np.sum)
            result_df.drop(columns = ['distance_details'], inplace = True)

            print(result_df)


            rec_player = closest_players[1]['player']
               
            
            print(f"Input Player: {self.player_name} (Team: {self.team})")
            print('Salary:')
            salary_input_player = self.player_salary(self.player_name)
            print(salary_input_player)
            
            ##self.plot_distance(closest_players)
            ##p_ids = [stats.reset_index()['PLAYER_ID'][closest_idx[i]] for i in range(len(closest_idx))]
            ##p_ids.append(self.player_id)
            ##self.plot_distance2(p_ids)

            
            print(f'\nRecommended Player: {rec_player}')
            print('Salary:')
            salary_rec_player = self.player_salary(rec_player)
            print(salary_rec_player)
            
            #print('-> Change in salary:')
            change_salary = self.change_salary(list(salary_input_player.iloc[0,1:]), list(salary_rec_player.iloc[0,1:]))
            #display(change_salary)
            
            #print('Salary Input Team:')
            team_salary = self.team_salary()
            #display(team_salary)
            
            #limit_salary = self.limit_salary_team(team_salary)
        
            #print('New Salary Input Team:')
            new_team_salary = self.new_team_salary(change_salary, team_salary)
            #display(new_team_salary)

            # Take with caution because also many players still have 0 salary
            print(f"Change in projected luxury tax: {[(luxury_tax(new_team_salary.iloc[0, i]) - luxury_tax(team_salary.iloc[0, i])) for i in range(3, team_salary.shape[1])]}")
            
            #visualize_capspace(team_salary.append(new_team_salary).append(limit_salary), 
            #                   ['Old Salary', f"New Salary by adding {rec_player}", 'Limit salary'],
            #                   self.team)

            ##visualize_capspace(team_salary.append(new_team_salary), 
            ##                   ['Old Salary', f"New Salary by adding {rec_player}"],
            ##                   self.team)
            
            return rec_player, result_df
        
        print("No data available for this player in the last season")
        pass
    
    
    def distance_comp(self, node, nodes, distance_measure, topN = 5): # adapt to 10 ???
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

        ## add cosine similarity
        # elif distance_measure == 'Cosine Similarity':
        
        topN_ids = np.argsort(distances)[: topN]
        #topN_dict = [{'player': players_data['player_names'][idx], 'distance': distances[idx]} for idx in topN_ids]  
#        print(distances[:topN + 1])
        return topN_ids[:topN + 1], np.sort(distances)[:topN], distances_detailed[topN_ids, :]
    
    def player_salary(self, rec_player):
        return players_salaries[players_salaries['player_names'] == rec_player]
    
    def change_salary(self, df_inputplayer, df_recplayer):
        # input - recommended
        change = [float(df_inputplayer[i]) - float(df_recplayer[i]) for i in range(1, len(df_inputplayer))] 
        return change
    
    def team_salary(self):
        # option with Commonplayerinfo (-> infos from this season (so, future))
        #data_rec_player = commonplayerinfo.CommonPlayerInfo(player_id=self.player_id) 
        # abb_team = list(data_rec_player.get_data_frames()[0]['TEAM_ABBREVIATION'])[0]
        
        abb_team = list(players_data[players_data['id'] == self.player_id]['team'])[0]
        return teams_salaries[teams_salaries['Abb'] == abb_team]
    
    def new_team_salary(self, change_salary, df_old_salary):
        df_new_salary = copy.copy(df_old_salary)
        for i in range(len(change_salary)):
            df_new_salary.iloc[0, 3+i] += change_salary[i]
        return df_new_salary
    
    def team_lastSeason(self):
        return list(players_data[players_data['id'] == self.player_id]['team'])[0]
    
    def limit_salary_team(self, team_salary):
        df_limit_salary = copy.copy(team_salary)
        for i in range(3, df_limit_salary.shape[1]):
            if df_limit_salary.iloc[0, i] > 0:
                pass
            else:
                df_limit_salary.iloc[0, i] *= 1.1 # may overdraw another 10 % -> oder via penalty function via betrag der negativität
        return df_limit_salary
    
    def plot_distance(self, dist_dict):
        #colors = ['blue', 'green', 'red']

        fig, ax = plt.subplots(figsize=(12, 5))
        x_values = [dist_dict[i]['distance'] for i in range(len(dist_dict))]
        x_values.append(0)
        y_values = (len(dist_dict)+1)*[0]
        colors = ['green']
        for i in range(len(y_values)-2):
            colors.append('blue')
        colors.append('red')
        
        names = [self.player_name]
        for i in range(len(dist_dict)):
            names.append(dist_dict[i]['player'])
        
        ax.scatter(x_values, y_values, color = colors, s = 70) #label = names
        #for i in range(len(x_values)):
        #    ax.scatter(x_values[i], y_values[i], label = names[i], color = colors[i], s = 70) #label = names
            
        ax.set(title = f'Recommendation system for {self.player_name}', xlabel = "Distance")
        ax.axes.get_yaxis().set_visible(False)
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        left_side = ax.spines["left"]
        left_side.set_visible(False)
        top_side = ax.spines["top"]
        top_side.set_visible(False)

       # plt.xticks(x, " ")
        
        #ax.legend(names)

        ax.annotate(self.player_name, (x_values[-1], y_values[-1]), xytext = (x_values[-1], y_values[-1] + 0.02), arrowprops = {'arrowstyle': '->'} )
        for i in range(len(dist_dict)):
            ax.annotate(dist_dict[i]['player'], (x_values[i], y_values[i]), xytext = (x_values[i], y_values[i] + 0.02*np.power(-1, i)), arrowprops = {'arrowstyle': '->'} )
    
        return plt.show()
    
    def plot_distance2(self, p_idxs):
        colors = ['green']
        for i in range(len(p_idxs)-2):
            colors.append('blue')
        colors.append('red')
            
        idxs = [(self.stats.index[self.stats['PLAYER_ID'] == p]).tolist()[0] for p in p_idxs]
        player_names = [list(players_data[players_data['id'] == p]['player_names'])[0] for p in p_idxs]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(title = f'Recommendation system for {self.player_name}')

        for i in range(len(p_idxs)):
            ax.scatter(self.stats.iloc[idxs[i], 5], self.stats.iloc[i,6], 
                       #label = players_data[players_data['id'] == r[i]], 
                       color = colors[i], s = 70)

            ax.annotate(player_names[i], 
                        (self.stats.iloc[idxs[i], 5], self.stats.iloc[i,6]), xytext = (self.stats.iloc[idxs[i], 5], self.stats.iloc[i,6] + 0.5*np.power(-1, i)), arrowprops = {'arrowstyle': '->'} ) 
    
        return plt.show()
    
    
    def get_maxs_teams(self):
        performance_teams = np.zeros((teams_data.shape[0], len(self.stats.columns)-5))
        for i in range(teams_data.shape[0]):
            team_abb = list(teams_data['abbreviation'])[i]
            start_five_team = list(starting_five(boxscores_20_21, team_abb, names = False).keys())
            data_team = pd.concat([self.stats[self.stats['PLAYER_ID'] == start_five_team[i]] for i in range(len(start_five_team))])
            performance_teams[i, :] = np.array(data_team.iloc[:,5:].sum(axis=0))

        maxs = np.amax(performance_teams, axis=0) 
        # oder: umdrehen bei import via - die column?
        col_min = ['PLAYER_AGE', 'PF', 'TOV']
        ind_min = [self.stats.columns.get_loc(i)-5 for i in col_min if i in self.stats.columns]
        #self.stats_agg[]

        for i in range(len(ind_min)):
            maxs[i] = np.amin(performance_teams[:,i])

        #maxs[0] = np.amin(performance_teams[:,0]) # PLAYER_AGE
        #maxs[19] = np.amin(performance_teams[:,19]) # PF
        #maxs[20] = np.amin(performance_teams[:,20]) # TOV

        maxs = maxs
        #display(maxs)

        return maxs
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def model_teams(self, input_player, agg_data_input_team):
        teams = {'20-21':['MIL', 'PHX', 'LAC', 'ATL'], '19-20':['LAL', 'MIA', 'DEN', 'BOS'], '18-19':['TOR', 'GSW', 'MIL', 'POR'], '17-18': ['GSW', 'CLE', 'HOU', 'BOS']} 
        team_ind = 0
        start_players_replaced_stats = []

        num_teams = sum([len(teams[x]) for x in teams if isinstance(teams[x], list)]) # count total number of model teams
        stats_teams = np.zeros((num_teams, len(self.cols_sel) - 8)) # not store values for player_id : team_abbreviation + GS, GP, MIN
        for i, season in enumerate(teams):
            seasons_past = seasons[seasons.index(season) - 3 : seasons.index(season)]   
            agg_data_seasons, _ = aggregate_data(players_stats, [f"20{season}" for season in seasons_past], self.w, self.cols_sel)
            
            for j in range(len(teams[season])):
                team = teams[season][j]
                #s_five = starting_five(boxscores_list[seasons.index(season)], team, names = False, current_season = False)
                #print(team, starting_five(boxscores_list[seasons.index(season)], team, names = True, current_season = False))
                s_five = playoff_player(season, team)
                #print(team, s_five)
                stats_players = []
                dist_to_input_player = []
                players_not_found = []
                for s_five_player in s_five.keys():
                    stats_player = agg_data_seasons[agg_data_seasons['PLAYER_ID'] == int(s_five_player)].iloc[:,5:] # with int() conversion: not good style
                    stats_players.append(stats_player)
                    _, distance, _ = self.distance_comp(input_player, stats_player.to_numpy(), self.distance_measure, topN = 2)
                    try:
                        dist_to_input_player.append(distance[0])
                    except IndexError: # e.g. rookies, players retired many years ago
                        #print(s_five_player, season, seasons_past, stats_player.to_numpy())
                        #print(players_stats[players_stats['PLAYER_ID'] == int(s_five_player)])
                        players_not_found.append(s_five_player)
                #except ValueError:
                #    print(s_five_player)
                    #print(agg_data_seasons[agg_data_seasons['PLAYER_ID'] == s_five_player])
                 #   pass # has to be adjusted ???
                for player in players_not_found:
                    ind_player = list(s_five.keys()).index(player)
                    avg_player = np.mean(pd.concat(stats_players).to_numpy(), axis = 0)
                    _, distance, _ = self.distance_comp(input_player, avg_player, self.distance_measure, topN = 2)
                    #dist_to_input_player.append(distance[0])
                    dist_to_input_player.insert(ind_player, distance[0])

                # remove closest player & save as possible target player -> via posiiton
                ind_same_position = [i for i in range(len(s_five)) if list(s_five.values())[i] == self.position]
                dist_relevant = [dist_to_input_player[i] for i in ind_same_position]
                start_player_replaced_pos = np.argmin(dist_relevant) # relevant index inside position group
                start_player_replaced = ind_same_position[start_player_replaced_pos] # relevant index inside entire starting five
                start_player_replaced_stats = stats_players.pop(start_player_replaced)
                start_players_replaced_stats.append(start_player_replaced_stats)

                # aggregate performance of remaining players of team
                data_team = np.abs(np.array(pd.concat(stats_players).sum(axis=0)))
                stats_teams[team_ind,:] = data_team #i+j
                team_ind += 1
                
        #print(stats_teams.shape, len(start_players_replaced_stats))
        
        # k-means, spectral clustering
        range_n_clusters = np.arange(2, stats_teams.shape[0]) # clusters from 2-15
        silhouette_avg = []
        Sum_of_squared_distances = []
        for num_clusters in range_n_clusters:
        
            # initialise kmeans
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(stats_teams)
            cluster_labels = kmeans.labels_
            
            # silhouette score
            silhouette_avg.append(silhouette_score(stats_teams, cluster_labels))                   
            Sum_of_squared_distances.append(kmeans.inertia_)

        '''
        plt.plot(range_n_clusters,silhouette_avg,'bx-')
        plt.xlabel('Values of K') 
        plt.ylabel('Silhouette score') 
        plt.title('Silhouette analysis For Optimal k')
        plt.show()

        plt.plot(range_n_clusters,Sum_of_squared_distances,'bx-')
        plt.xlabel('Values of K') 
        plt.ylabel('Sum of squared distances/Inertia') 
        plt.title('Elbow Method For Optimal k')
        plt.show()
        '''
        k_opt = range_n_clusters[np.argmax(silhouette_avg)] # change ???
        kmeans = KMeans(n_clusters = k_opt).fit(stats_teams)
        model_teams_labels = kmeans.labels_
        cluster_pred = kmeans.predict(data_team.reshape(1, -1))[0]
        #print('Cluster:', cluster_pred)
        #print('Teams:', model_teams_labels)
        ind_role_teams = [i for i, label in enumerate(model_teams_labels) if label == cluster_pred]
        
        repl_player_role_teams = [start_players_replaced_stats[i] for i in ind_role_teams]
        repl_player_agg = pd.concat(repl_player_role_teams).mean(axis=0)
        #print(np.array(repl_player_agg), repl_player_agg.shape)

        return repl_player_agg

# aggregated player: best one


seasons = ['14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21']
boxscores_list = [boxscores_14_15, boxscores_15_16, boxscores_16_17, boxscores_17_18, boxscores_18_19, boxscores_19_20, boxscores_20_21] 

# API calls abschalten
# Spieler mit meisten Minuten ; Berechnung von den einzelnen 5 Datafeames pro Team außerhalb von Klasse machen & dann Importen  


# add input of certain player -> remove closest player in position -> then sum up team's data & cluster
# 5th player -> leader secondary (management bank spieler: wichtig)

# take starting five of year with good result & aggregate their performance of their previous three seasons
# bucks: champion 2020-2021 -> 17-18, 18-19, 19-20
# top 4 teams of last 5 years as models or top 8 teams of last 3 years?
# get boxscores -> starting five (-> CLUSTERN: dann auswahl möglich von durchschnittlichen attributen je cluster) & remove player of position of replacing player **see in data/season_prediction**
# get aggregated performances of these players over past 3 seasons  -> aggregate team performance
# -> compare similarity of this team to the team at end

def playoff_player(season, team_abb, names = False):
    season_mapping = {'20-21': '2020', '19-20': '2019', '18-19': '2018', '17-18': '2017'}
    season = season_mapping[season]
    positions = {'F': 2, 'C': 1, 'G': 2}
    data_team = playoffs_boxscores[playoffs_boxscores['SEASON'] == season]
    data_team = data_team[data_team['TEAM_ABBREVIATION'] == team_abb].loc[:, ['PLAYER_ID', 'MIN']]
    potential_players = list(data_team['PLAYER_ID'].dropna().unique())

    positions_players = [list(playoffs_players[playoffs_players['PLAYER_ID'] == p_id]['POS'])[0] for p_id in potential_players]

    players_sec = list()

    for player in potential_players:
        data_player = data_team[data_team['PLAYER_ID'] == player]
        sec_played = data_player['MIN'].sum()
        #players_min.append({'player_id': player, 'sec': min_played})
        players_sec.append(sec_played)
    start_five = {}
    for i in range(len(positions)):
        pos = list(positions.keys())[i]
        n_player = positions[pos]
        ind_players = [i for i, e in enumerate(positions_players) if e == pos]
        sec_players = [players_sec[i] for i in ind_players]
        ind_most_played = list(np.argsort(sec_players))[-n_player:]
        #print(pos, n_player, ind_players, sec_players, ind_most_played)
        for i in ind_most_played:
            player_id = potential_players[ind_players[i]]
            if names:
                name = list(playoffs_players[playoffs_players['PLAYER_ID'] == player_id]['NAME'])[0]
                start_five[name] = pos 

            else:
                start_five[player_id] = pos                               
    return start_five



# Exemplary execution
if __name__ == "__main__":
    w = [7/10, 2/10, 1/10]
    cols_sel = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
    stats_agg, stats_agg_notTransformed = aggregate_data(players_stats, ['2020-21', '2019-20', '2018-19'], w, cols_sel) # [0, 1, 2, 3, 4, 5, 6, 7, 8])
    data_emb, emb, _, _, _ = embeddings('spectral', stats_agg, stats_agg_notTransformed, dim=3)
    sample_recommendation = RecommendationEngine(stats_agg, "LeBron James", 'Fit', 'L2', w, cols_sel).recommend()

    #print(commonplayerinfo.CommonPlayerInfo(201939).get_data_frames()[0]['SEASON_EXP', 'HEIGHT'])
    




    #teams = {'20-21':['MIL', 'PHX', 'LAC', 'ATL'], '19-20':['LAL', 'MIA', 'DEN', 'BOS'], '18-19':['TOR', 'GSW', 'MIL', 'POR'], '17-18': ['GSW', 'CLE', 'HOU', 'BOS']} 
    #for season in teams.keys():
    #    teams_season = teams[season]
    #    for team in teams_season:
    #        playoff_player(season, team, names = True)


    #print(players_stats[players_stats['PLAYER_ID'] == '2585']) # ???

    #Index(['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION',
    #       'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
    #       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
    #       'BLK', 'TOV', 'PF', 'PTS'],

    # deduct the current season