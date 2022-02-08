from rapm_toolbox import *

data_path = "./data_dsp/"
dummy_file = "dummy.pm.csv"

# load in data_dummy
cols_to_drop = ['stint', 'HOME_PTS', 'AWAY_PTS', 'HOME_PM_100', 'Unnamed: 0']
cols = list(pd.read_csv(data_path + dummy_file, nrows = 1))
data_dummy = pd.read_csv(data_path + dummy_file, dtype={'GAME_ID':str}, usecols =[i for i in cols if ~(i in cols_to_drop)])

print("Loaded dummy data")
model = RidgeCV(alphas=[1e-3, 1e-1, 1e1, 1e3, 1e4], normalize=True, cv=5)

print("Start estimation")
model, player_ids = estimate_model(data_dummy=data_dummy, ids_start=2, col_y="HOME_PM", model=model)

print("Finished estimation")

# dump model and safe scores
pickle.dump(model, open(data_path + "return_model_ridge_no_cv", 'wb'))
scores = show_scores_player(coef=model.coef_, series_id_player=player_ids)
scores.to_csv(data_path + "return_player_scores.csv")
