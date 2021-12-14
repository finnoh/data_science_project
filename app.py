import dash
import wikipediaapi
import requests
from PIL import Image
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from src.get_data import get_clean_player_data, get_team_image
from src.utils_dash import _player_selector, _team_selector, _link_team_website, _team_full_name

import dash_bootstrap_components as dbc

app = dash.Dash(__name__, title="NBA GM", external_stylesheets=[dbc.themes.LUX])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()
team_selector = _team_selector()

# APP ELEMENTS
offcanvas = html.Div(
    [
        dbc.Button("Further Team Information", id="teamselect-open-offcanvas", n_clicks=0, color='light'),
        dbc.Offcanvas([html.Div(
            html.P(id='teamselect-output-wiki'
                   )),

            dbc.Button(id="teamselect-link-button",
                       target="_blank",
                       color="info", external_link=True)],
            backdrop=True,
            scrollable=True,
            id="offcanvas",
            is_open=False,
            placement='top'
        ),
    ]
)

left_jumbotron = dbc.Col(
    html.Div(
        [
            html.H2("Select", className="display-3"),
            dcc.Dropdown(
                id='teamselect-dropdown',
                options=[{'label': team,
                          'value': team_selector.iloc[
                              i, 1]} for i, team in
                         enumerate(
                             team_selector.label.unique())],
                value='ATL'
            ),
            html.Hr(className="my-2"),
            html.Div(
                [html.Img(id='teamselect-image'),
                 offcanvas],
                style={'margin': 'auto'}
            )

        ],
        className="h-100 p-5 bg-light border rounded-3",
    ),
    md=4,
)

middle_jumbotron = dbc.Col(
    html.Div(
        [html.Div(
            [html.H2(id='teamselect-output-container', className="display-3")]),
            html.Hr(className="my-2")
        ],
        className="h-100 p-5 bg-light border rounded-3",
    ),
    md=8,
)

jumbotron = dbc.Row(
    [left_jumbotron, middle_jumbotron],
    className="align-items-md-stretch",
)

# APP LAYOUT

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Player Bio', value='tab-1', children=[
            html.H1(children='NBA GM'),
            html.Div([html.Div(
                [html.Img(id='playerselect-image')]
            )], style={'width': '49%'}
            ), html.Div(
                [dcc.Dropdown(
                    id='playerselect-dropdown',
                    options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in
                             enumerate(player_selector.label.unique())],
                    placeholder='Select a Player',
                    value=203500
                )], style={'width': '20%'}
            ),
            html.Div(id='playerselect-output-container'),
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')],
                     style={'width': '49%', 'display': 'inline-block'})

        ]),
        dcc.Tab(label='Performance', value='tab-2', children=[dbc.Container([
            dash_table.DataTable(
                id='playerselect-table'
            ),
            dbc.Alert(id='tbl_out')]),

            dbc.Container([
                dcc.Graph(id='playerselect-graph1')
            ])]),

        dcc.Tab(label='Team', value='tab-3', children=[jumbotron]
                )])
])


# APP CALLBACKS

@app.callback(
    Output('playerselect-output-container', 'children'),
    Input('playerselect-dropdown', 'value'))
def update_output(value):
    return f'Player has the ID: {value}'


@app.callback(
    Output('teamselect-output-container', 'children'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return _team_full_name(value)


@app.callback(
    Output('offcanvas', 'title'),
    Input('teamselect-dropdown', 'value'))
def update_output(value):
    return _team_full_name(value)


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
    [Output('playerselect-table', 'data'),
     Output('playerselect-table', 'columns'),
     Output('playerselect-graph1', 'figure')],

    [Input('playerselect-dropdown', 'value')])
def update_player(value):
    # make api call
    df = get_clean_player_data(player_id=value)

    # get objects
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    # get figure
    fig = px.line(df, x="SEASON_ID", y="PTS")
    fig.update_layout(transition_duration=500)

    return data, columns, fig


@app.callback(
    dash.dependencies.Output('playerselect-image', 'src'),
    [dash.dependencies.Input('playerselect-dropdown', 'value')])
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


if __name__ == '__main__':
    app.run_server(debug=True)
