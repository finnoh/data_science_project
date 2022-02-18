from hotzone import hotzone
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
from src.tabs import player, team, recommendation, mincer_tab

app = dash.Dash(__name__, title="NBA GM", external_stylesheets=[dbc.themes.LUX])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# APP LAYOUT
app.layout = html.Div(children=[
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Player', value='tab-1', children=[
            player.jumbotron_player, html.Hr(className="my-2"), player.top_players, player.draft_pick_performance
        ]),
        dcc.Tab(label='Team', value='tab-5', children=[team.jumbotron
                                                       ]),
        dcc.Tab(label='Recommendation', value='tab-3', children=[
            html.H2(children='Trade Assist', className="display-3"),
            html.Div([recommendation.cards_rec, recommendation.dim_red])
        ]),
        dcc.Tab(label="Salary", value='tab-4', children=[mincer_tab.mincer])
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

    print([weight, height, draft])

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
    Output('dimreduction-graph1', 'figure'),
    [Input('dimreduction-dropdown', 'value')])
def get_emb(value):
    players_stats, _, positions, data_names, player_stats = recommmendation_engine.embeddings(value)
    name_emb = {'spectral': 'Sepectral Embedding', 'tsne': 'TSNE', 'umap': 'UMAP', 'pca': 'PCA'}
    #    player_stats.positions = positions
    #    player_stats.head()
    fig = px.scatter(players_stats, x="embedding_1", y="embedding_2", color=positions, hover_name=data_names,
                     hover_data={'embedding_1': False,
                                 'embedding_2': False,
                                 'Position': positions,
                                 'Age': player_stats['PLAYER_AGE'],
                                 'Points': player_stats['GP'],
                                 '3P PCT': (':.3f', player_stats['FG3_PCT']),
                                 'Assists': (':.3f', player_stats['AST']),
                                 'Rebounds': (':.3f', player_stats['REB'])
                                 },
                     labels={"embedding_1": "Embedding Dimension 1", "embedding_2": "Embedding Dimension 2"},
                     title=f"{name_emb[str(value)]} representation of NBA players")
    fig.update_layout(transition_duration=500, template='simple_white')
    return fig


@app.callback(
    Output('teamRec-starting5-dropdown', 'options'),
    [Input('teamRec-select-dropdown', 'value')])
def get_starting_five(value):
    players_team = recommmendation_engine.starting_five(value, names=True)
    return [{'label': i, 'value': i} for i in players_team.keys()]


@app.callback(
    Output('playerRep-image', 'src'),
    [Input('teamRec-starting5-dropdown', 'value')])
def update_image_repPlayer(value):
    player_id = list(player_data[player_data['player_names'] == value]['id'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"


@app.callback(
    Output('teamRep-image', 'src'),
    [Input('teamRec-select-dropdown', 'value')])
def update_image_repTeam(value):
    return f"http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/{value.lower()}.png"


@app.callback(
    Output('teamRec-player-dropdown', 'children'),
    [Input('teamRec-starting5-dropdown', 'value'), Input('recommendation-type', 'value')])
def selected_player(rep_player, rec_type):
    data_emb, emb, _, _, _ = recommmendation_engine.embeddings('umap')
    sample_recommendation = recommmendation_engine.RecommendationEngine(data_emb, rep_player, emb,
                                                                        rec_type)  # 'Similar'
    r = sample_recommendation.recommend()
    return r


# @app.callback(
#    Output('teamRec-player-name', 'children'),
#    Input('teamRec-player-dropdown', 'children'))
# def print_recommended_player(value):
#    return f"{value}"

@app.callback(
    Output('playerRec-image', 'src'),
    Input('teamRec-player-dropdown', 'children'))
def update_image_recPlayer(children):
    player_id = list(player_data[player_data['player_names'] == children]['id'])[0]
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(player_id)}.png"

@app.callback(
    Output('teamselect-capspace-graph', 'figure'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return recommmendation_engine.visualize_capspace_team_plotly(value)


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
