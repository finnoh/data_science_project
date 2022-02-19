import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamdashboardbylastngames


def _player_selector():
    """ Helper for selecting a player with a selector

    :return tmp3: dict, label and value pairs for selector
    """
    # get_players returns a list of dictionaries, each representing a player.
    tmp = pd.DataFrame(players.get_active_players())  # just return the ids
    tmp2 = tmp[['full_name', 'id']]
    tmp3 = tmp2.rename(columns={"full_name": "label", "id": "value"})
    return tmp3


def _team_selector():
    """

    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'GAME_ID': str})
    tmp = df[['full_name', 'abbreviation']]
    tmp2 = tmp.rename(columns={"full_name": "label", "abbreviation": "value"})
    return tmp2


def _team_full_name(abbreviation):
    """

    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'GAME_ID': str})
    tmp = df[df['abbreviation'] == abbreviation]['full_name'].values

    return tmp[0].lower()


def _player_full_name(player_id):
    """

    :return:
    """
    df = pd.read_csv("players_data.csv", dtype={'id': str})
    player_name = df[df['id'] == player_id]['player_names'].values[0]
    player_pos = df[df['id'] == player_id]['position'].values[0]

    return player_name, player_pos


def _link_team_website(abbreviation):
    """

    :param abbreviation:
    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'TEAM_ID': str})
    tmp = df[df['abbreviation'] == abbreviation]['nickname'].values

    return tmp[0].lower()


def _get_team_id(abbreviation):
    """

    :param abbreviation:
    :return:
    """
    df = pd.read_csv("data/data_assets/teams.csv", dtype={'TEAM_ID': str})
    team_id = df[df['abbreviation'] == abbreviation]['TEAM_ID'].values

    return team_id[0]


def _get_mvp_id_team(team_id, season: str = '2020-21'):
    """

    :param abbreviation:
    :return:
    """

    df = pd.read_csv("playercareerstats_activeplayers.csv", dtype={'TEAM_ID': str, 'PLAYER_ID': str})

    subset = df[(df['TEAM_ID'] == team_id) & (df['SEASON_ID'] == season)]
    mvp = subset[subset['PTS'] == subset['PTS'].max()]

    mvp_id = mvp['PLAYER_ID'].values[0]
    mvp_age = int(mvp['PLAYER_AGE'].values[0])
    mvp_gp = mvp['GP'].values[0]
    mvp_pts, mvp_ast, mvp_reb = np.round(mvp['PTS'].values[0] / mvp_gp, 1), np.round(mvp['AST'].values[0] / mvp_gp, 1), np.round(mvp['REB'].values[0] / mvp_gp, 1)
    url_image = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{str(mvp_id)}.png"

    mvp_data = [mvp_id, mvp_age, mvp_pts, mvp_ast, mvp_reb, mvp_gp]

    return mvp_data, url_image


def _mvp_descr_builder(mvp_name, mvp_position, mvp_data):
    """

    :param mvp_name:
    :param mvp_position:
    :param mvp_data:
    :return:
    """

    rand = np.random.randint(low=0, high=6)
    rand_excellent = np.random.randint(low=0, high=6)
    rand_star = np.random.randint(low=0, high=6)

    word_excellent = ['phenomenal', 'excellent', 'stellar', 'amazing', 'outstanding']
    word_star = ['icon', 'all-star caliber', 'franchise-player', 'star-player', 'our mvp']

    if mvp_position == "G":
        position = "Guard"
    elif mvp_position == "F":
        position = "Forward"
    else:
        position = "Center"

    if rand == 0:
        sentence = f"Our {str(mvp_data[1])} year old {position} {mvp_name} scored {word_excellent[rand_excellent]} {str(mvp_data[2])} points last season."
    elif rand == 1:
        sentence = f"{mvp_name}'s ({mvp_position}) stats last season were {word_excellent[rand_excellent]} {str(mvp_data[2])} points while contributing an average {str(mvp_data[3])} assists per game."
    elif rand == 3:
        sentence = f"{mvp_data[5]} games last season make our {word_star[rand_star]} {position}, {mvp_name} one of our most busy players. At an age of {str(mvp_data[1])} he scored {str(mvp_data[2])} points while securing {str(mvp_data[4])} boards per game."
    elif rand == 4:
        sentence = f"{str(mvp_data[2])} points, {str(mvp_data[3])} assists and {str(mvp_data[4])} per game. Those are the stats of our  {str(mvp_data[1])} year old {word_star[rand_star]} {mvp_name}."
    else:
        sentence = f"At {str(mvp_data[1])} years of age, {mvp_name} leads the team with {str(mvp_data[2])} point per game. The {word_excellent[rand_excellent]} {position} also dishes out {str(mvp_data[3])} assists a night. He played {mvp_data[5]} games last season."

    return sentence

# def _team_lastngames(team_id, n, season: str = "2021-22"):
#     """
#
#     :param team_id:
#     :param n:
#     :param season:
#     :return:
#     """
#     call = teamdashboardbylastngames.TeamDashboardByLastNGames(team_id=str(team_id), last_n_games=n, season=season)
#     df = pd.concat(call.get_data_frames())
#     print(df)
#     return df

def draw_plotly_court(fig, fig_width=600, margins=10):
    import numpy as np

    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ),

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ),
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ),

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return fig