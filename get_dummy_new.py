from src_dsp.rapm_toolbox import *
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = "./data_dsp/"

# read in schedule and extract games of specific season for estimation
df_schedule = pd.read_csv(data_path + "schedule.csv", dtype={'GAME_ID':str})
list_game_id = df_schedule[df_schedule['SEASON_ID'] == 22020]['GAME_ID'].to_list()

print("Read in data")
subs = pd.read_csv(data_path + "subs.csv", dtype={'GAME_ID':str})
starters = pd.read_csv(data_path + "starters.csv", dtype={'GAME_ID':str})
data = pd.read_csv(data_path + "preprocessing.csv", dtype={'GAME_ID':str})

# subset
subs = subs[subs['GAME_ID'].isin(list_game_id)]
data = data[data['GAME_ID'].isin(list_game_id)]
starters = starters[starters['GAME_ID'].isin(list_game_id)]

print("Get on court")
df_on_court = get_on_court(starters=starters, subs=subs)
df_on_court.to_csv(data_path + "on_court.csv")

print("Get Dummy")
dummy, dummy_ids = stints_to_dummy_simple(data=data, df_on_court=df_on_court)
dummy.to_csv(data_path + "dummy_pm_simple.csv")
dummy_ids.to_csv(data_path + "dummy_pm_simple_ids.csv")
