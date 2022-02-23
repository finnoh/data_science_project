from dash import dcc, html
import dash_bootstrap_components as dbc

text = '''
**Welcome to the NBA dashboard by Finn Höner and Tim-Moritz Bündert!**

We are glad that you are visiting our dashboard which was designed as part of the course Data Science Project (DS500) at the University of Tuebingen.
As basketball is not only our passion but also offers a very relevant business case while being well-accessible using data, this topic presents a interesting and holistic data project application.

Importantly, the point of time of this dashboard is at the end of the season 2020/2021 and before the season 2021/2022, so the current one.
Accordingly, the roster changes which happended in the mean time are not present in the dashboard, but deliberately left out. This of course also entails any performance statistics of the current season.

It features in total four tabs, which are explained in more detail below.

## Tab 1: Player
This Tab presents both general and more detailed information on NBA players. This includes career performance, draft statistics and salary information. 
Beyond this exploratory data anaylsis, we present an own calculated player score, which tries to estimate a players performance in a holistic fashion. 
Besides this performance measure, we predict a player's salary based on his in-game performance which provides a baseline to compare his real salary too. 
This then can be used to determine whether a player is currently under- or over-priced.

## Tab 2: Team
Here, general and more detailed information on all 30, current, NBA teams are presented. It also embeds statistics on the development of a team's salary cap and the player's on it's roster.

## Tab 3: Recommendation
This tab features a complex recommendation engine, which assist NBA general managers with finding substitutes for current players on their roster.
This algorithm aids with finding trade targets, who are as similar as possible to the selected player or who would complement the roster in an optimal way.
The presented recommendations are supplemented with more detailed information on the selected attributes and the calculated player score (see *Player tab*) and information on a player's salary following our Mincer analysis.
It is also possible, to analyze the recommendation visually by looking at the NBA's players in a low-dimensional embedding based on the selected features such as age or points per game.
After selecting a trade target, we predict how the team's performance in the upcoming season changes after this trade has been conducted.

## Tab 4: Season prediction
This tab presents our season prediction model, which is based on a bayesian linear regression model. 
Choose your team and replace your current players with other players from all around the league - without any constraints. The "simulation" tab presents
the outcome of a whole simulated season as well as simulated table rank outcomes for the team at hand. The "validation" tab shows the prediction performance of 
the model on the real game results of the 2021/22 season, up to mid february.


# References
The corresponding data along with the names and images displayed in the dashboard are not our property. Rather, the following references apply:
* [NBA](https://www.nba.com/termsofuse), in particular 
    - Images of teams: http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/....png
    - Images of players: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/....png
* [NBA_API](https://github.com/swar/nba_api)
* Team salaries from https://www.spotrac.com/nba/cap/
* Player salaries from https://hoopshype.com/salaries/players/
* [NBA2k ratings](https://2kmtcentral.com/21/players/collection/)
* [RAPM Scores](http://nbashotcharts.com/home)
* [NBA Logo](https://logosmarken.com/nba-logo/)
'''


welcome_tab = html.Div([
                        html.H2(children='Welcome to the NBA Dashboard', className="display-3"),
                        dbc.Row([dbc.Col([dcc.Markdown(text)], width=7),
                                 dbc.Col(html.Div(html.Img(src="/assets/NBA.png", title="Uni Tübingen x NBA"),
                                                  style = {'textAlign': 'center', 'position': 'fixed', 'margin-left': '15%'}
                                                  ), width = 5)])
                    ], style = {'margin-left': '20px'})

                    #style = {'width': '100%', 'display': 'flex', 