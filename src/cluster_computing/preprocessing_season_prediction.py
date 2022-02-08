from sklearn.model_selection import train_test_split
import itertools

df_schedule = pd.read_csv("./data/season_prediction/schedule.csv"),
df_boxscores = pd.read_csv("./data/season_prediction/boxscores.csv")
scroes = pd.read_csv("scores.csv")

# select seasons
seasons = [22020, 22019]
df_schedule = df_schedule[df_schedule['SEASON_ID'].isin(seasons)]

games = df_schedule['GAME_ID'].unique()
df_boxscores = df_boxscores[df_boxscores['GAME_ID'].isin(games)]

# merge schedule and boxscores
df = pd.merge(df_schedule, df_boxscores, how='left', on=["GAME_ID", 'TEAM_ID'], suffixes=("_team", "_player"))
df['home_game'] = df['MATCHUP'].apply(lambda x: x.find('@') == -1)

# merge scores and df
df = pd.merge(df, scores, how='left', left_on='PLAYER_ID', right_on='playerId')

# create sixth man
sixth_man = pd.DataFrame(df[df['START_POSITION'].isna()].groupby(['TEAM_ID', 'GAME_ID'])['RAPM'].aggregate('mean')).reset_index().drop_duplicates()
sixth_man['START_POSITION'] = "SIXTH_MAN"
df_sixth = pd.merge(sixth_man, df[['GAME_ID', 'TEAM_ID', 'home_game']], on=['GAME_ID', 'TEAM_ID'])
df_sixth = df_sixth.drop_duplicates()

# calculate bench depth
df['BENCH'] = df['START_POSITION'].isna()
bench_depth = pd.DataFrame(df.groupby(['GAME_ID', 'TEAM_ID'])['BENCH'].sum()).reset_index().drop_duplicates()
bench_depth = bench_depth.rename(columns={'BENCH':'BENCH_DEPTH'})
df = pd.merge(df, bench_depth, how='left')

# mark bench depth
df.loc[df['home_game'], 'BENCH_DEPTH_TEAM'] = 'BENCH_DEPTH_HOME'
df.loc[~df['home_game'], 'BENCH_DEPTH_TEAM'] = 'BENCH_DEPTH_AWAY'

# append to dataframe
df = df.append(df_sixth)

# filter for Starters only
df = df[~df['START_POSITION'].isna()]
df2 = df

# create counts
half = int(df.shape[0] / 2)
repetitions_half = int(half / 6)

# first step for role player variable
sixth_home = ['SIXTH_HOME_1'] * repetitions_half
guard_home = ['GUARD_HOME_1', 'GUARD_HOME_2'] * repetitions_half
forward_home = ['FORWARD_HOME_1', 'FORWARD_HOME_2'] * repetitions_half
center_home = ['CENTER_HOME_1'] * repetitions_half

sixth_away = ['SIXTH_AWAY_1'] * repetitions_half
guard_away = ['GUARD_AWAY_1', 'GUARD_AWAY_2'] * repetitions_half
forward_away = ['FORWARD_AWAY_1', 'FORWARD_AWAY_2'] * repetitions_half
center_away = ['CENTER_AWAY_1'] * repetitions_half

# create player role
player_role = list(itertools.chain(sixth_home, guard_home, forward_home, center_home,
                                   sixth_away, guard_away, forward_away, center_away))

# sort values and insert the player role
df = df.sort_values(['home_game', 'START_POSITION'], ascending=False)
df['PLAYER_COURT_ROLE'] = np.asarray(player_role)

# subset dataframes
df_sub = df[['PLUS_MINUS', 'home_game', 'TEAM_ID','TEAM_ABBREVIATION_team','GAME_ID', 'PLAYER_ID', 'playerName', 'PLAYER_COURT_ROLE', 'RAPM', 'BENCH_DEPTH_TEAM', 'BENCH_DEPTH']]
df_reg = df_sub[['PLUS_MINUS', 'home_game', 'TEAM_ID','GAME_ID', 'PLAYER_ID', 'PLAYER_COURT_ROLE', 'RAPM', 'BENCH_DEPTH_TEAM', 'BENCH_DEPTH']]

sub_bench = df_reg[['GAME_ID', 'BENCH_DEPTH_TEAM', 'BENCH_DEPTH']].drop_duplicates()
sub_bench = sub_bench.pivot('GAME_ID', 'BENCH_DEPTH_TEAM', 'BENCH_DEPTH').reset_index()

# create df model frame
df_reg_scores = df_reg[['GAME_ID', 'PLUS_MINUS', 'home_game']]
wide = df_reg.pivot('GAME_ID', 'PLAYER_COURT_ROLE', 'RAPM').reset_index()
df_model = pd.merge(wide, df_reg_scores)
df_model = pd.merge(df_model, sub_bench, how='left', on='GAME_ID')

df_model = df_model[['home_game', 'GUARD_HOME_1', 'GUARD_HOME_2', 'FORWARD_HOME_1', 'FORWARD_HOME_2', 'CENTER_HOME_1', 'SIXTH_HOME_1', 'BENCH_DEPTH_HOME', 'GUARD_AWAY_1', 'GUARD_AWAY_2', 'FORWARD_AWAY_1', 'FORWARD_AWAY_2', 'CENTER_AWAY_1', 'SIXTH_AWAY_1', 'BENCH_DEPTH_AWAY', 'PLUS_MINUS']]

# drop duplicates and keep only observations from HT perspective
df_model = df_model.drop_duplicates()
df_model = df_model.dropna()
df_model = df_model[df_model['home_game']]

# seperate into labels and features
y = df_model['PLUS_MINUS'].to_numpy()
features_names = ['GUARD_HOME_1', 'GUARD_HOME_2', 'FORWARD_HOME_1', 'FORWARD_HOME_2', 'CENTER_HOME_1', 'SIXTH_HOME_1', 'BENCH_DEPTH_HOME', 'GUARD_AWAY_1', 'GUARD_AWAY_2', 'FORWARD_AWAY_1', 'FORWARD_AWAY_2', 'CENTER_AWAY_1', 'SIXTH_AWAY_1', 'BENCH_DEPTH_AWAY']
X = df_model[features_names].to_numpy()

# get bool for na rows
na_rows = ~np.isnan(X).any(axis=1)

# remove na rows
X = X[na_rows]
y = y[na_rows]

# train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
