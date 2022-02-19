from rapm_toolbox import *

data_load = load_data_pbp(season_ids=[22020, 22019, 22018, 22017, 22016])
data_load.to_csv("pbp_16_20.csv")
