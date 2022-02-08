from rapm_toolbox import *

# set data path
data_path = "./data_dsp/"

# read in the play by play data
data = pd.read_csv(data_path + "pbp_safe_data.csv", dtype={'GAME_ID':str})

# create bools
bool_sub = (data['EVENTMSGTYPE'] == 8).to_numpy() # substitutions
bool_tech = ((data['EVENTMSGTYPE'] == 6) & (data['EVENTMSGACTIONTYPE'].isin([10, 11, 16, 18, 25]))).to_numpy() # technical fouls
bool_eject_tech = ((data['EVENTMSGTYPE'] == 11) & (data['EVENTMSGACTIONTYPE'] == 1)).to_numpy() # ejct 2nd technical fouls
bool_ingame_plays = ~(bool_sub | bool_tech | bool_eject_tech)

# drop unnecessary columns
data = data.drop(['WCTIMESTRING', 'PCTIMESTRING', 'NEUTRALDESCRIPTION',
                  'PLAYER3_TEAM_NICKNAME', 'PLAYER3_TEAM_ABBREVIATION', 'PLAYER3_TEAM_CITY',
                 'PLAYER2_TEAM_NICKNAME', 'PLAYER2_TEAM_ABBREVIATION', 'PLAYER2_TEAM_CITY',
                 'PLAYER1_TEAM_NICKNAME', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER1_TEAM_CITY',
                 'VIDEO_AVAILABLE_FLAG', 'SCOREMARGIN'], axis=1)

# preprocessing
data = data.groupby(['GAME_ID']).apply(lambda x: create_stint(x))
data = get_score(data)
data = estimate_possessions(data=data, bool_ingame_plays=bool_ingame_plays)
data = estimate_pm_100(data=data)

# safe the preprocessing file
data.to_csv(data_path + "preprocessing.csv")
print("Safed preprocessing.csv")

# get starters and roster
starters, roster = get_roster_and_starters(data)

# get starting lineups, subs and on court players
starting_lineups = get_starting_lineup(starters)
subs = get_all_subs(data=data)
court = get_on_court(data=data, starters=starters, sub=subs)

# safe objects
starting_lineups.to_csv(data_path + "starting_lineups.csv")
subs.to_csv(data_path + "subs.csv")
roster.to_csv(data_path + "roster.csv")
court.to_csv(data_path + "court.csv")

print("Safed court and roster objects")

# create data_stints
col_score = ['HOME_PTS', 'AWAY_PTS', 'HOME_PM', 'HOME_PM_100', 'EST_POSSESSIONS']
data_stints = merge_stint_pts(data=data, court_data=court, col_score=col_score)

# safe
data_stints.to_csv(data_path + "data_stints.csv")

print("Safed data_stints.csv")

# create dummy matrix and get player_ids
dummy_pm, player_ids = stints_to_dummy_unique_pm(data_stints=data_stints)

#safe
dummy_pm.to_csv(data_path + "dummy.pm.csv")
player_ids.to_csv(data_path + "player_ids.csv")

print("Safed dummy_pm.csv")
