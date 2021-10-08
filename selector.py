import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd


df = pd.read_csv("players_tmp.csv")

cols = ['full_name', 'Image_URL', 'TEAM_ABBREVIATION']

df = df.loc[:, cols]
df.drop_duplicates(subset='full_name', inplace=True)

import dash
import dash_table
import pandas as pd

app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns]
)

if __name__ == '__main__':
    app.run_server(debug=True)
