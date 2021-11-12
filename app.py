import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from src.get_data import get_clean_player_data
from src.utils_dash import _player_selector

import dash_bootstrap_components as dbc

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# SETUP STATIC DATA
player_selector = _player_selector()

# APP LAYOUT

app.layout = html.Div(children=[
    html.H1(children='NBA GM'),

    dcc.Dropdown(
            id='playerselect-dropdown',
            options=[{'label': player, 'value': player_selector.iloc[i, 1]} for i, player in enumerate(player_selector.label.unique())],
            placeholder='Select a Player',
            value=203500
        ),

    html.Div(id='playerselect-output-container'),

    html.Div(
        html.Img(id='playerselect-image')
    ),

    dbc.Container([
        dbc.Label('Click a cell in the table:'),
        dash_table.DataTable(
            id='playerselect-table'
        ),
        dbc.Alert(id='tbl_out')]),

    dbc.Container([
        dcc.Graph(id='playerselect-graph1')
])


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


if __name__ == '__main__':
    app.run_server(debug=True)
