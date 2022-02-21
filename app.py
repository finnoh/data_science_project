from mplcursors import HoverMode
from hotzone import hotzone
import dash
import wikipediaapi
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from src.get_data import get_clean_player_data
from src.utils_dash import _player_selector
from hotzone import hotzone
import copy
import dash
import wikipediaapi
import requests
from PIL import Image
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
from src.utils_dash import _player_selector, _team_selector, _link_team_website, _team_full_name, _get_team_id, \
    _get_mvp_id_team, _player_full_name, _mvp_descr_builder, draw_plotly_court
import dash
import wikipediaapi
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from src.get_data import get_clean_player_data, get_player_score
from src.utils_dash import _player_selector
import recommmendation_engine as recommmendation_engine

import dash_bootstrap_components as dbc
from src.tabs import player, team, recommendation

import flask
server = flask.Flask(__name__)
app = dash.Dash(__name__, title="NBA GM", external_stylesheets=[dbc.themes.LUX], server = server)

from dash.dependencies import Input, Output, State
from src.utils_dash import _player_selector, _player_full_name, _team_selector, _team_full_name, _get_team_id, \
    _get_mvp_id_team, _mvp_descr_builder, draw_plotly_court, _link_team_website
from src.mincer import *
import dash
import wikipediaapi
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from src.get_data import *
import recommmendation_engine
import dash_bootstrap_components as dbc
from src.tabs import player, team, recommendation, mincer_tab, welcome
#import json
import io


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


# SETUP STATIC DATA
player_selector = _player_selector()
team_selector = _team_selector()
player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()
players_stats = recommmendation_engine.get_players_stats()
boxscores_20_21 = recommmendation_engine.get_boxscores('20_21')



# APP LAYOUT
app.layout = html.Div(children=[
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label="Welcome", value='tab-1', children=[welcome.welcome_tab]),
        dcc.Tab(label='Player', value='tab-2', children=[
            player.jumbotron_player, html.Hr(className="my-2"), player.top_players, player.draft_pick_performance
        ]),
        dcc.Tab(label='Team', value='tab-3', children=[team.jumbotron
                                                       ]),
        dcc.Tab(label='Recommendation', value='tab-4', children =[recommendation.recommendation_tab]),
        dcc.Tab(label="Salary", value='tab-5', children=[mincer_tab.mincer])
    ], colors={
        "border": "white",
        "primary": "#17408b",
        "background": "white"})
])


# SETUP FOR CALLBACKS
player_selector = _player_selector()
team_selector = _team_selector()
player_data = recommmendation_engine.get_players_data()
team_data = recommmendation_engine.get_teams_data()


# APP CALLBACKS

####### Tab 0: Welcome

####### Tab 1: Player
@app.callback(
    [Output('playerselect-output-container', 'children'),
     Output('playerselect-name-container', 'children')],
    Input('playerselect-dropdown', 'value'))
def update_output(value):
    player_name, _ = _player_full_name(str(value))
    return [f'Player has the ID: {value}'], player_name


@app.callback(
    [Output('teamselect-output-container', 'children'),
     Output('offcanvas', 'title')],
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return _team_full_name(value), _team_full_name(value)

####### Tab 2: Team
@app.callback(
    [dash.dependencies.Output('teamselect-mvp-image', 'src'),
     dash.dependencies.Output('teamselect-mvp-descr', 'children'),
     dash.dependencies.Output('teamselect-mvp-name', 'children')],
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def update_output(value):
    team_id = _get_team_id(value)
    mvp_data, url_image = _get_mvp_id_team(team_id=team_id, season='2020-21')
    mvp_name, mvp_pos = _player_full_name(player_id=mvp_data[0])
    descr = _mvp_descr_builder(mvp_name=mvp_name, mvp_position=mvp_pos, mvp_data=mvp_data)
    return url_image, descr, mvp_name


@app.callback(
    Output('teamselect-link-button', 'children'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    full_name = _team_full_name(value)
    return f'Visit the {full_name}\'s website.'


@app.callback(
    Output('teamselect-link-button', 'href'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    nickname = _link_team_website(value)
    return f"https://www.nba.com/{nickname}/"


@app.callback(
    Output('teamSel-image', 'src'),
    [Input('playerselect-dropdown', 'value')])
def update_image_selTeam(value):
    team_abb = list(player_data[player_data['id'] == value]['team'])[0]
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{team_abb.lower()}.png"



@app.callback(
    [Output('playerselect-topplayer', 'data'),
    Output('playerselect-topplayer', 'columns')],
    [Input('playerselect-dropdown', 'value')])
def topplayer_table(value):
    df = get_all_player_score()
    data = df.to_dict('records')
    cols = df.columns.tolist()
    columns = [{"name": i, "id": i} for i in cols]
    return data, columns

@app.callback(
    Output("graph", "figure"),
    [Input("pick", "value")])
def pick_hist(value):
    df = get_all_player_score()

    # pick range
    pick_lower = value[0]
    pick_upper = value[1]

    # range of picks
    picks = np.arange(pick_lower, pick_upper + 1)

    # subset
    tmp = df[df['Draft Number'].isin(picks)]

    # figure
    fig = px.histogram(tmp, x="Top %", nbins=60, range_x=[1, 60], marginal="rug", hover_data=tmp.columns)
    return fig




@app.callback(
    [Output('playerselect-table', 'data'),
     Output('playerselect-table', 'columns'),
     Output('playerselect-graph1', 'figure'),
     Output('playerselect-score', 'children'),
     Output('playerselect-graph2', 'figure'),
     Output('playerselect-draft', 'children'),
     Output('playerselect-bio', 'children')],
    [Input('playerselect-dropdown', 'value')])
def update_player(value):
    # make api call
    data_all = get_clean_player_data(player_id=value)
    cols = ['SEASON_ID', 'PLAYER_AGE', 'GP', 'MIN', 'PTS', 'AST', 'REB', 'BLK']
    df = data_all[cols]

    weight = data_all['WEIGHT_KG'].values[0]
    height = data_all['HEIGHT_METER'].values[0]
    draft = data_all['DRAFT_NUMBER'].values[0]
    draft_year = data_all['DRAFT_YEAR'].values[0]
    draft_round = data_all['DRAFT_ROUND'].values[0]
    previous = data_all['LAST_AFFILIATION'].values[0]

    body = f'{weight} kg, {height} m'
    drafted = f'At {draft} in round {draft_round} - {draft_year} \n ({previous})'
    player_score = get_player_score(player_id=value)

    # get objects
    columns = [{"name": i, "id": i} for i in cols]
    data = df.to_dict('records')

    df_season = get_season_data(player_id=value)
    df_salary = get_player_salary(player_id=value)

    # get figure
    fig = px.line(df_season, x="SEASON", y="coef_perc_rank", range_x=[2016, 2020], range_y=[0, 100])
    fig.update_layout(transition_duration=500, template="simple_white")

    fig2 = px.line(df_salary, x="SEASON", y="value", range_x=[2016, 2024])
    fig2.update_layout(transition_duration=500, template="simple_white")

    return data, columns, fig, [f'Overall Top {player_score} %'], fig2, body, drafted


@app.callback(
    Output('playerselect-image', 'src'),
    [Input('playerselect-dropdown', 'value')])
def update_image_src(value):
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(value)}.png"


@app.callback(
    dash.dependencies.Output('teamselect-image', 'src'),
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def get_team_image(value, season: str = '2021-22'):
    """
    :param team_abb:
    :return:
    """
    url = f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"
    return url


@app.callback(
    dash.dependencies.Output('teamselect-output-wiki', 'children'),
    [dash.dependencies.Input('teamselect-dropdown', 'value')])
def _team_wiki_summary(value):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    full_name = _team_full_name(value)
    page_py = wiki_wiki.page(full_name)
    return page_py.summary


@app.callback(
    Output("offcanvas", "is_open"),
    Input("teamselect-open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output('playerselect-output-container-wiki', 'children'),
    [Input('playerselect-dropdown', 'value')])
def _player_wiki_summary(value):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    df = player_selector
    name = list(df[df['value'] == value]['label'])[0]
    page_py = wiki_wiki.page(name)
    if page_py.exists():
        return page_py.summary
    else:
        return f"No Wikipedia page found for {str(name)}"


@app.callback(
    Output('hotzone-graph', 'figure'),
    [Input('playerselect-dropdown', 'value')])
def hotzone_graph(value):
    shots = hotzone(value)
    fig = px.scatter(x=shots["LOC_X"], y=shots["LOC_Y"], color=shots['SHOT_MADE_FLAG'].astype(str),
                     width=1200, height=1000, opacity=0.5)
    # fig.update_layout(transition_duration=500, template='simple_white')
    # fig = px.density_heatmap(shots, x="LOC_X", y="LOC_Y", z="SHOT_MADE_FLAG", histfunc="avg", width=1200, height=1000)

    # fig = fig.update_layout(template="simple_white")
    fig = draw_plotly_court(fig=fig, fig_width=600)
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, xaxis_visible=False, xaxis_showticklabels=False)
    return fig


@app.callback(
    Output('teamselect-capspace-graph', 'figure'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return recommmendation_engine.visualize_capspace_team_plotly(value)

####### Tab 3: Recommendation
@app.callback(
    Output("infocanvas", "is_open"),
    Input("rec-infocanvas", "n_clicks"),
    [State("infocanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('mincer-output-rec', 'data'),
    [Input('mincer-rec-dropdown', 'value'), Input('teamRec-select-dropdown', 'value')]
)
def update_output(modelname, team):
    # train, test split
    X_train, y_train, df_train, X_test, y_test, df_test, X, y, df = select_features()

    # select model and tuning
    model, param_grid = select_model_grid(model_name=modelname)

    # fit the model, score and predict on whole data set
    model_fitted = wrapper_tune_fit(X_train=X_train, y_train=y_train, model=model, param_grid=param_grid)
    score = score_model(X_test=X_test, y_test=y_test, model=model_fitted)
    prediction, model_fitted = fit_predict_full(X_train, y_train, X_test, model=model_fitted)

    # create dataframe for plot and the plot itself
    df_plot = create_plot_dataset(prediction=prediction, y=y_test, df=df_test)

    df_salary = df_plot[['id', 'Predicted']].astype({"id": int, "Predicted": float})

    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=False)
    players = list(players_team.keys())

    results_player = {} # {p: round(list(df_salary[df_salary['id'] == p]['Predicted'])[0], 2) for p in players}
    for p in players:
        try:
            results_player[p] = round(list(df_salary[df_salary['id'] == p]['Predicted'])[0], 2)
        except:
            results_player[p] = 0.0

    return results_player 


@app.callback(
    [Output('playerRep-image_1', 'src'),
     Output('playerRep-str_1', 'children'),
     Output('playerRep-price_1', 'children')],
    [Input('teamRec-select-dropdown', 'value'),
     Input('mincer-output-rec', 'data')])
def get_starting_five(team, predicted_salaries):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[3]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]

    pred_salary = predicted_salaries[str(player_id)]
    if pred_salary > 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'green'
    elif pred_salary == 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'black'
    else:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'red'

    result_salary = html.Div([html.Div('Mincer Analysis:', style = {'margin-right': '5px'}),
                              html.Div(pred_salary, style = {"color": c})], style = {'display':'flex'})

    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})", result_salary

@app.callback(
    [Output('playerRep-image_2', 'src'),
     Output('playerRep-str_2', 'children'),
     Output('playerRep-price_2', 'children')],
    [Input('teamRec-select-dropdown', 'value'),
     Input('mincer-output-rec', 'data')])
def get_starting_five(team, predicted_salaries):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[4]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    pred_salary = predicted_salaries[str(player_id)]
    if pred_salary > 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'green'
    elif pred_salary == 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'black'
    else:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'red'

    result_salary = html.Div([html.Div('Mincer Analysis:', style = {'margin-right': '5px'}),
                              html.Div(pred_salary, style = {"color": c})], style = {'display':'flex'})
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})", result_salary

@app.callback(
    [Output('playerRep-image_3', 'src'),
     Output('playerRep-str_3', 'children'),
     Output('playerRep-price_3', 'children')],
    [Input('teamRec-select-dropdown', 'value'),
     Input('mincer-output-rec', 'data')])
def get_starting_five(team, predicted_salaries):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[0]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    pred_salary = predicted_salaries[str(player_id)]
    if pred_salary > 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'green'
    elif pred_salary == 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'black'
    else:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'red'

    result_salary = html.Div([html.Div('Mincer Analysis:', style = {'margin-right': '5px'}),
                              html.Div(pred_salary, style = {"color": c})], style = {'display':'flex'})
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})", result_salary

@app.callback(
    [Output('playerRep-image_4', 'src'),
     Output('playerRep-str_4', 'children'),
     Output('playerRep-price_4', 'children')],
    [Input('teamRec-select-dropdown', 'value'),
     Input('mincer-output-rec', 'data')])
def get_starting_five(team, predicted_salaries):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[1]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    pred_salary = predicted_salaries[str(player_id)]
    if pred_salary > 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'green'
    elif pred_salary == 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'black'
    else:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'red'

    result_salary = html.Div([html.Div('Mincer Analysis:', style = {'margin-right': '5px'}),
                              html.Div(pred_salary, style = {"color": c})], style = {'display':'flex'})
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})", result_salary


@app.callback(
    [Output('playerRep-image_5', 'src'),
     Output('playerRep-str_5', 'children'),
     Output('playerRep-price_5', 'children')],
    [Input('teamRec-select-dropdown', 'value'),
     Input('mincer-output-rec', 'data')])
def get_starting_five(team, predicted_salaries):
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    player_name = list(players_team.keys())[2]
    player_id = list(player_data[player_data['player_names'] == player_name]['id'])[0]
    player_pos = list(player_data[player_data['player_names'] == player_name]['position'])[0]
    pred_salary = predicted_salaries[str(player_id)]
    if pred_salary > 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'green'
    elif pred_salary == 0.0:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'black'
    else:
        pred_salary = '${:,.2f}'.format(pred_salary)
        c = 'red'

    result_salary = html.Div([html.Div('Mincer Analysis:', style = {'margin-right': '5px'}),
                              html.Div(pred_salary, style = {"color": c})], style = {'display':'flex'})
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png", f"{player_name} ({player_pos})", result_salary


#    Output('playerRep-image', 'src'),
#    [Input('teamRec-starting5-dropdown', 'value')])
#def update_image_repPlayer(value):
#    player_id = list(player_data[player_data['player_names'] == value]['id'])[0]
#    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"


@app.callback(
    Output('teamRep-image', 'src'),
    [Input('teamRec-select-dropdown', 'value')])
def update_image_repTeam(value):
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"


@app.callback(
    [Output('teamRec-player-dropdown', 'children'),
    #Output('playerRec-table', 'data'),
    Output('playerRec-table', 'children'),
    Output('players-recommended', 'data'),
    Output('btn_1', 'n_clicks'), Output('btn_2', 'n_clicks'), Output('btn_3', 'n_clicks'), Output('btn_4', 'n_clicks'), Output('btn_5', 'n_clicks'), Output('pos_img', 'data'), 
    Output('alert-triggered', 'data'),
    Output('alert-features-triggered', 'data'),
    Output('playerRec-stats', 'data')],
    [Input('teamRec-select-dropdown', 'value'), State('recommendation-type', 'value'), State('recommendation-distance', 'value'),
     State("checklist-all-details", "value"), State("checklist-off-details", "value"), State("checklist-off2-details", "value"), State("checklist-def-details", "value"),
     Input('btn_1', 'n_clicks'), Input('btn_2', 'n_clicks'), Input('btn_3', 'n_clicks'), Input('btn_4', 'n_clicks'), Input('btn_5', 'n_clicks')],
     State('weight1', 'value'), State('weight2', 'value'), State('weight3', 'value'), Input('alert-triggered', 'data'), Input('alert-features-triggered', 'data'), Input('mincer-rec-dropdown', 'value')
     )
def selected_player(team, rec_type, dist_m, cols_all, cols_off, cols_off2, cols_def, b1, b2, b3, b4, b5, w1, w2, w3, weights_error, features_error, mincer_option):  
    players_team = recommmendation_engine.starting_five(boxscores_20_21, team, names=True)
    weights = [w1/100, w2/100, w3/100] #[7/10, 2/10, 1/10]

    if (rec_type == 'Fit') & len(set(['Playmaking', 'Athleticism', 'Score']).intersection(set(cols_all))) > 0:
        features_error = True
        print(features_error)
        return 0, 0, 0, None, None, None, None, None, 0, False, features_error, 0 # data, columns
    
    else:
        features_error = False

    if w1 + w2 + w3 != 100: # add??
        weights_error = True
        #print('Error:', weights)
    else:
        weights_error = False

    if b1 is not None:
        rep_player = list(players_team.keys())[3]
        b1 = None
        pos = 1
    elif b2 is not None:
        rep_player = list(players_team.keys())[4]
        b2 = None
        pos = 2
    elif b3 is not None:
        rep_player = list(players_team.keys())[0]
        b3 = None
        pos = 3
    elif b4 is not None:
        rep_player = list(players_team.keys())[1]
        b4 = None
        pos = 4
    elif b5 is not None:
        rep_player = list(players_team.keys())[2]  
        b5 = None
        pos = 5
    else:
        rep_player = list(players_team.keys())[3] 
        pos = 1
        
    cols = cols_all + cols_off + cols_off2 + cols_def

    #if 'PLAYER_AGE' in cols:
    #    sel_col = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE' 'GP', 'GS', 'MIN']
    #    cols.remove('PLAYER_AGE')
    #else:
    sel_col = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'GS', 'MIN']

    stats_agg, _ = recommmendation_engine.aggregate_data(players_stats, w = weights, cols = sel_col+cols)
    #data_emb, emb, _, _, _ = recommmendation_engine.embeddings('umap', stats_agg, stats_agg_notTransformed)
    sample_recommendation = recommmendation_engine.RecommendationEngine(stats_agg, rep_player, rec_type, distance_measure = dist_m, w = weights, cols_sel = sel_col+cols) # 'Similar'
    r, result_table = sample_recommendation.recommend()

    ## Mincer values
    X_train, y_train, df_train, X_test, y_test, df_test, X, y, df = select_features()
    model, param_grid = select_model_grid(model_name=mincer_option)  # select model and tuning

    # fit the model, score and predict on whole data set
    model_fitted = wrapper_tune_fit(X_train=X_train, y_train=y_train, model=model, param_grid=param_grid)
    score = score_model(X_test=X_test, y_test=y_test, model=model_fitted)
    prediction, model_fitted = fit_predict_full(X_train, y_train, X_test, model=model_fitted)

    # create dataframe for plot and the plot itself
    df_plot = create_plot_dataset(prediction=prediction, y=y_test, df=df_test)

    df_salary = df_plot[['id', 'Predicted']].astype({"id": int, "Predicted": float})

    players = list(result_table['player'])
    ids = [list(player_data[player_data['player_names'] == player]['id'])[0] for player in players]

    results_player = [] # {p: round(list(df_salary[df_salary['id'] == p]['Predicted'])[0], 2) for p in players}
    for p in ids:
        try:
            results_player.append(round(list(df_salary[df_salary['id'] == p]['Predicted'])[0], 2))
        except:
            results_player.append(0.0)


    # formatting the table
    result_table.rename({'luxury_tax': 'Luxury Tax'}, axis=1, inplace=True)
    if 'HEIGHT' in cols:
        result_table.rename({'HEIGHT': 'Height'}, axis=1, inplace=True)
    if 'WEIGHT' in cols:
        result_table.rename({'WEIGHT': 'Weight'}, axis=1, inplace=True)
    if 'EXPERIENCE' in cols:
        result_table.rename({'EXPERIENCE': 'Experience'}, axis=1, inplace=True)
    if 'PLAYER_AGE' in cols:
        result_table.rename({'PLAYER_AGE': 'Age'}, axis=1, inplace=True)
        result_table['Age'] = [list(players_stats[(players_stats['PLAYER_ID'] == p) & (players_stats['SEASON_ID'] == '2020-21')]['PLAYER_AGE'])[0] for p in ids]
    
    result_table.distance = result_table.distance.round(2)
    result_table['id'] = result_table.index 
    result_table.insert(2, "Priced", results_player) ## ???

    columns = [{"name": i, "id": i} for i in result_table.columns]
    data = result_table.to_dict('records')

    tooltip_columns = {'player': 'Player name', 
                       'distance': 'Overall distance over all selected attributes',
                       'Age': 'Age of player',
                       'Priced': 'Over-/under-priced according to Mincer model', 
                       'Luxury Tax': 'Sum of luxury tax (in $) over next four seasons',
                       'FGM': 'Field Goals Made',
                       'FG3_PCT': '3-point Percentage',
                       'FGA': 'Field Goals Attempted',
                       'FG3M': '3-point Shots Made',
                       'FTM': 'Free Throws Made',
                       'FG3A': '3-point Shots Attempted',
                       'FG_PCT': 'Field Goals Percentage',
                       'FT_PCT': 'Free Throws Percentage',
                       'TOV': 'Turnovers',
                       'FTA': 'Free throws attempted',
                       'AST': 'Assists',
                       'PTS': 'Points',
                       'OREB': 'Offensive rebounds',
                       'DREB': 'Defensive rebounds',
                       'PF': 'Personal fouls',
                       'STL': 'Steals',
                       'BLK': 'Blocked Shots',
                       'REB': 'Total Rebounds',
                       'Height': 'Height (in cm)',
                       'Weight': 'Weight (in kg)',
                       'Experience': 'Years in the league',
                       'Score': 'Player score computed on stints',
                       'Athleticism': 'XXXX',
                       'Playmaking': 'XXXX'}

    dt = dash_table.DataTable(
            data = data,
            columns=[{'name': i, 'id': i} for i in result_table.columns if i != 'id'],
            style_data_conditional=recommendation.highlight_max_col(result_table),
            tooltip_header={i:tooltip_columns[i] for i in list(result_table.columns) if i != 'id'},
            sort_action="native",
            style_cell={'minWidth': '100px'}
        )

    players_plot = list(result_table['player'])
    #players_plot.insert(0, rep_player)

    #print(result_table)
    #print(players_plot)
    #print()

    return r, dt, players_plot, b1, b2, b3, b4, b5, pos, weights_error, features_error, stats_agg.to_json() # data, columns

#stats_agg, stats_agg_notTransformed = aggregate_data(players_stats, [7/10, 2/10, 1/10], ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT'])


# @app.callback(
#    Output('teamRec-player-name', 'children'),
#    Input('teamRec-player-dropdown', 'children'))
# def print_recommended_player(value):
#    return f"{value}"


@app.callback(
    Output("alert-weights", "is_open"),
    [Input('btn_1', 'n_clicks'), Input('btn_2', 'n_clicks'), Input('btn_3', 'n_clicks'), Input('btn_4', 'n_clicks'), Input('btn_5', 'n_clicks')],
    [State("alert-weights", "is_open"), State("alert-triggered", "data")],
)
def toggle_alert(b1, b2, b3, b4, b5, triggered, is_open):
    if triggered:
        return not is_open
    return is_open

@app.callback(
    Output("alert-features", "is_open"),
    [Input('btn_1', 'n_clicks'), Input('btn_2', 'n_clicks'), Input('btn_3', 'n_clicks'), Input('btn_4', 'n_clicks'), Input('btn_5', 'n_clicks'),
     Input('alert-features-triggered', 'data')],
    [State("alert-features", "is_open")],
)
def toggle_alert(b1, b2, b3, b4, b5, triggered, is_open):
    return triggered


@app.callback(
    [Output('playerRec-image_1', 'src'),Output('playerRec-image_2', 'src'),Output('playerRec-image_3', 'src'),Output('playerRec-image_4', 'src'),Output('playerRec-image_5', 'src'),
     Output('playerRec-caption_1', 'children'),Output('playerRec-caption_2', 'children'),Output('playerRec-caption_3', 'children'),Output('playerRec-caption_4', 'children'),Output('playerRec-caption_5', 'children')],
    [Input('teamRec-player-dropdown', 'children'),
     Input('pos_img', 'data')])
def update_image_recPlayer(player, pos):
    player_id = list(player_data[player_data['player_names'] == player]['id'])[0]
    player_team = list(player_data[player_data['player_names'] == player]['team'])[0]
    player_pos = list(player_data[player_data['player_names'] == player]['position'])[0]

    pos = int(pos)
    img1 = ''
    img2 = ''
    img3 = ''
    img4 = ''
    img5 = ''

    cap1 = ''
    cap2 = ''
    cap3 = ''
    cap4 = ''
    cap5 = ''

    output_str = f'{player} ({player_team}, {player_pos})' 

    if pos == 1:
        img1 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap1 = output_str
    elif pos == 2:
        img2 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap2 = output_str
    elif pos == 3:
        img3 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap3 = output_str
    elif pos == 4:
        img4 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap4 = output_str
    else:
        img5 = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
        cap5 = output_str

    return img1, img2, img3, img4, img5, cap1, cap2, cap3, cap4, cap5


# @app.callback(
#    Output('teamRec-player-name', 'children'),
#    Input('teamRec-player-dropdown', 'children'))
# def print_recommended_player(value):
#    return f"{value}"

'''
@app.callback(
    Output('playerRec-image', 'src'),
    Input('teamRec-player-dropdown', 'children'))
def update_image_recPlayer(children):
    player_id = list(player_data[player_data['player_names'] == children]['id'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"
'''



@app.callback(
    [Output("checklist-all-details", "value"), Output("checklist-off", "value"), Output("checklist-def", "value")],
    [Input("checklist-all", "value")],
    [State("checklist-all-details", "options"), State("checklist-off", "options"), State("checklist-def", "options")],
)
def select_all_none(all_group, all, off, defense):
    if len(all_group) == 0:
        return [], [], [] #all, off, off2, defense
    else:
        attributes_all = [option["value"] for option in all]
        attributes_off = [option["value"] for option in off]
        attributes_def = [option["value"] for option in defense]  
        return attributes_all, attributes_off, attributes_def


@app.callback(
    [Output("checklist-off-details", "value"), Output("checklist-off2-details", "value")],
    [Input("checklist-off", "value")],
    [State("checklist-off-details", "options"), State("checklist-off2-details", "options")],
)
def select_all_none(all_selected, options1, options2):
    if len(all_selected) == 0:
        return [], []
    else:
        attributes_1 = []
        attributes_2 = []
        if 'Off' in all_selected:
            off_cols1 = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM'] 
            off_cols2 = ['FTA', 'FT_PCT', 'AST', 'PTS', 'TOV', 'OREB']
            for option in options1:
                if option['value'] in off_cols1:
                    attributes_1.append(option["value"])
                elif option['value'] in off_cols2:
                    attributes_2.append(option["value"])
            for option in options2:
                if option['value'] in off_cols1:
                    attributes_1.append(option["value"])
                elif option['value'] in off_cols2:
                    attributes_2.append(option["value"])
            #attributes = attributes = [option["value"] for option in options if option['value'] in def_cols]
        return list(set(attributes_1)), list(set(attributes_2))

@app.callback(
    Output("checklist-def-details", "value"),
    [Input("checklist-def", "value")],
    [State("checklist-def-details", "options")],
)
def select_all_none(all_selected, options):
    if len(all_selected) == 0:
        return []
    else:
        attributes = []
        if 'Def' in all_selected:
            def_cols = ['OREB', 'DREB', 'REB', 'STL', 'BLK', 'TOV', 'PF']
            for option in options:
                if option['value'] in def_cols:
                    attributes.append(option["value"])
            #attributes = attributes = [option["value"] for option in options if option['value'] in def_cols]
        return list(set(attributes))

'''
@app.callback(
    Output("checklist-columns", "value"),
    [Input("checklist-all", "value"), Input("checklist-off", "value"), Input("checklist-def", "value")],
    [State("checklist-all-details", "options"), State("checklist-off-details", "options"), State("checklist-def-details", "options")],
)
def select_all_none(all_group, off_group, def_group, all_cols, off_cols, def_cols):
    print(all_group, off_group, def_group, all_cols, off_cols, def_cols)
    if len(all_selected) == 0:
        return []
    else:
        attributes = []
        if 'All' in all_selected:
            for option in options:
                attributes.append(option["value"])
        if 'Off' in all_selected:
            off_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'AST', 'PTS']
            for option in options:
                if option['value'] in off_cols:
                    attributes.append(option["value"])
            #attributes = [option["value"] for option in options if option['value'] in off_cols]
        if 'Def' in all_selected:
            def_cols = ['OREB', 'DREB', 'REB', 'STL', 'BLK', 'TOV', 'PF']
            for option in options:
                if option['value'] in def_cols:
                    attributes.append(option["value"])
            #attributes = attributes = [option["value"] for option in options if option['value'] in def_cols]
    
        return list(set(attributes))


@app.callback(
    Output("checklist-columns", "value"),
    [Input("checklist-allColumns", "value")],
    [State("checklist-columns", "options")],
)
def select_all_none(all_selected, options):
    if len(all_selected) == 0:
        return []
    else:
        attributes = []
        if 'All' in all_selected:
            for option in options:
                attributes.append(option["value"])
        if 'Off' in all_selected:
            off_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'AST', 'PTS']
            for option in options:
                if option['value'] in off_cols:
                    attributes.append(option["value"])
            #attributes = [option["value"] for option in options if option['value'] in off_cols]
        if 'Def' in all_selected:
            def_cols = ['OREB', 'DREB', 'REB', 'STL', 'BLK', 'TOV', 'PF']
            for option in options:
                if option['value'] in def_cols:
                    attributes.append(option["value"])
            #attributes = attributes = [option["value"] for option in options if option['value'] in def_cols]
    
        return list(set(attributes))
'''


@app.callback(
    Output('rec-dimreduction-graph1', 'figure'),
    [Input('rec-dimreduction-type', 'value'), Input('rec-dimreduction-dim', 'value'), Input('players-recommended', 'data')])
def get_emb(dim_type, dim, players):
    stats_agg, stats_agg_notTransformed = recommmendation_engine.aggregate_data(players_stats, w = [7/10, 2/10, 1/10]) 
    players_stats_emb, _, positions, data_names, player_stats = recommmendation_engine.embeddings(dim_type, stats_agg, stats_agg_notTransformed, int(dim)) # player
    name_emb = {'spectral': 'Sepectral Embedding', 'tsne': 'TSNE', 'umap': 'UMAP', 'pca': 'PCA'}

    #print('Callback', player_stats.head())
    labels = copy.deepcopy(list(positions))
    for i, player in enumerate(players):
        ind_player = list(player_data.index[player_data['player_names'] == player])[0]
        if i == 0:
            labels[ind_player] = player
        else:
            labels[ind_player] = 'Recommendations'

    #print('Here:', players)

    color_discrete_map = {'G': 'rgb(144,132,132)', 'F': 'rgb(203,197,197)', 'C': 'rgb(101,85,85)', 'Recommendations': 'rgb(66,171,59)', players[0]: 'rgb(250,49,69)'}

    if int(dim) == 2:
        fig = px.scatter(players_stats_emb, x="embedding_1", y="embedding_2", color = labels, hover_name = data_names, 
                        #color_discrete_sequence=["red", "green", "blue", "orange", "black"],
                        color_discrete_map=color_discrete_map,
                        #hover_data={'embedding_1':False, 
                        #            'embedding_2':False, 
                        #            'Position': positions,
                        #            'Age': player_stats['PLAYER_AGE'],
                        #            'Points': (':.3f', player_stats['PTS']),
                        #            '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                        #            'Assists': (':.3f', player_stats['AST']),
                        #            'Rebounds': (':.3f', player_stats['REB'])
                        #            },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} Representation of NBA players")
        fig.update_layout(legend_title_text = '<b>Positions / Players</b>')
        fig.update_layout(transition_duration=500, template='simple_white')
        fig.update_traces(hoverinfo='none', hovertemplate='')

    
    if int(dim) == 3:
        fig = px.scatter_3d(players_stats_emb, x="embedding_1", y="embedding_2", z = "embedding_3", color = labels, #hover_name = data_names, 
                        #color_discrete_sequence=["red", "green", "blue", "orange", "black"],
                        color_discrete_map=color_discrete_map,
                        #hover_data={'embedding_1':False, 
                        #            'embedding_2':False, 
                        #            'embedding_3':False, 
                        #            'Position': positions,
                        #            'Age': player_stats['PLAYER_AGE'],
                        #            'Points': (':.3f', player_stats['PTS']),
                        #            '3P PCT': (':.3f', player_stats['FG3_PCT']), 
                        #            'Assists': (':.3f', player_stats['AST']),
                        #            'Rebounds': (':.3f', player_stats['REB'])
                        #            },
                        labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2", "embedding_3": "Embedding Dimension 2"}, title=f"{name_emb[str(dim_type)]} representation of NBA players")
        fig.update_layout(legend_title_text = '<b>Positions / Players</b>')
        fig.update_layout(transition_duration=500, template='simple_white')
        fig.update_traces(hoverinfo='none', hovertemplate='')

    return fig


@app.callback(
    Output("graph-rec-tooltip", "show"),
    Output("graph-rec-tooltip", "bbox"),
    Output("graph-rec-tooltip", "children"),
    [Input('rec-dimreduction-graph1', "hoverData"), Input('playerRec-stats', 'data')]
)
def display_hover(hoverData, df):

    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    #print(hoverData)

    df = pd.read_json(df)

    name = pt['hovertext']
    player_id = list(player_data[player_data['player_names'] == name]['id'])[0]

    df_row = df[df['PLAYER_ID'] == player_id] #df.iloc[num]
    img_src = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"

    age = list(players_stats[(players_stats['PLAYER_ID'] == player_id) & (players_stats['SEASON_ID'] == '2020-21')]['PLAYER_AGE'])[0]
    points = df_row['PTS'].iloc[0]
    P3PCT = df_row['FG3_PCT'].iloc[0]
    Assists = df_row['AST'].iloc[0]
    Rebounds = df_row['REB'].iloc[0]

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.P(f"{name}", style={"color": "darkblue"}),
            html.P(f"{int(age)} years old", style={"color": "black"}),
            html.P(f"Points: {np.round((points), 2)}", style={"color": "black"}),
            html.P(f"3P-PCT: {np.round((P3PCT), 2)}", style={"color": "black"}),
            html.P(f"Assists: {np.round((Assists), 2)}", style={"color": "black"}),
            html.P(f"Rebounds: {np.round((Rebounds), 2)}", style={"color": "black"}),
        ], style={'width': '200px'})
    ]

    return True, bbox, children


#### Tab 4: Mincer

@app.callback([
    Output('mincer-output-container', 'children'), Output('mincer-output-graph', 'figure')],
    [Input('mincer-model-dropdown', 'value'), Input('mincer-log-switch', 'on')]
)
def update_output(modelname, log):

    # train, test split
    X_train, y_train, df_train, X_test, y_test, df_test, X, y, df = select_features()

    # select model and tuning
    model, param_grid = select_model_grid(model_name=modelname)

    # fit the model, score and predict on whole data set
    model_fitted = wrapper_tune_fit(X_train=X_train, y_train=y_train, model=model, param_grid=param_grid)
    score = score_model(X_test=X_test, y_test=y_test, model=model_fitted)
    prediction, model_fitted = fit_predict_full(X_train, y_train, X_test, model=model_fitted)

    # create dataframe for plot and the plot itself
    df_plot = create_plot_dataset(prediction=prediction, y=y_test, df=df_test)
    fig = plot_mincer(df_plot=df_plot, logarithm=log)

    return f"Coefficient of determination: {np.round(score * 100, 2)} %", fig

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input('mincer-output-graph', "hoverData"),
)
def display_hover(hoverData):

    if hoverData is None:
        return False, no_update, no_update

    df_plot = pd.read_csv("./data/tmp/mincer_plot.csv")
    #df_plot['TOP%'] = df_plot['coef_perc_rank'].rank(perc=True, ascending=False)

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df_plot.iloc[num]
    img_src = get_player_image(int(df_row['id']))
    name = df_row['DISPLAY_FIRST_LAST']
    age = df_row['PLAYER_AGE']
    salary = df_row['Salary']
    diff = df_row['Difference']
    rank = df_row['coef_perc_rank']

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.P(f"{name}", style={"color": "darkblue"}),
            html.P(f"{int(age)} years old", style={"color": "black"}),
            html.P(f"{np.round((salary / 1000000), 2)} Mil.", style={"color": "black"}),
            html.P(f"Diff: {np.round((diff / 1000000), 2)} Mil.", style={"color": "black"}),
            html.P(f"{np.round(rank, 2)}", style={"color": "black"})
        ], style={'width': '200px'})
    ]

    return True, bbox, children


if __name__ == '__main__':
    app.run_server(debug=True)