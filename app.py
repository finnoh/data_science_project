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

import dash_bootstrap_components as dbc

app = dash.Dash(__name__, title="NBA GM")

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()

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
            html.Div(children=[html.Div(id='playerselect-output-container-wiki')], style={'width': '49%', 'display': 'inline-block'})

        ]),
        dcc.Tab(label='Performance', value='tab-2', children=[dbc.Container([
            dash_table.DataTable(
                id='playerselect-table'
            ),
            dbc.Alert(id='tbl_out')]),

            dbc.Container([
                dcc.Graph(id='playerselect-graph1')
            ])]),
        dcc.Tab(label='Tab Three', value='tab-3', children=html.H1("Yet Another Page"))])
])


# APP CALLBACKS

@app.callback(
    Output('playerselect-output-container', 'children'),
    Input('playerselect-dropdown', 'value'))
def update_output(value):
    return f'Player has the ID: {value}'

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
    dash.dependencies.Output('playerselect-output-container-wiki', 'children'),
    [dash.dependencies.Input('playerselect-dropdown', 'value')])
def _player_wiki_summary(value):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(value)
    return f"https://simple.wikipedia.org/wiki/Chris_Paul"

if __name__ == '__main__':
    app.run_server(debug=True)
