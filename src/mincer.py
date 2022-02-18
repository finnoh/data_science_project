import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import plotly.express as px

def select_features(features: list=['RAPM', 'POSITION_Center', 'POSITION_Center-Forward', 'POSITION_Forward',
                'POSITION_Forward-Center', 'POSITION_Forward-Guard', 'POSITION_Guard',
                'POSITION_Guard-Forward', 'HEIGHT_METER', 'WEIGHT_KG',
                'GP', 'GS', 'PTS', 'AST', 'OREB', 'DREB', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'ROOKIE', 'PLAYER_AGE', 'BMI', 'FROM_US',
                'PLAYER_AGE_2', 'DRAFT_NUMBER', 'Undrafted']):
    """

    :param features: list of features to use - from selector
    :return:
    """

    df_model = pd.read_csv("./data/data_assets/mincer_data.csv")

    features = ['RAPM', 'POSITION_Center', 'POSITION_Center-Forward', 'POSITION_Forward',
                'POSITION_Forward-Center', 'POSITION_Forward-Guard', 'POSITION_Guard',
                'POSITION_Guard-Forward', 'HEIGHT_METER', 'WEIGHT_KG',
                'GP', 'GS', 'PTS', 'AST', 'OREB', 'DREB', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'ROOKIE', 'PLAYER_AGE', 'BMI', 'FROM_US',
                'PLAYER_AGE_2', 'DRAFT_NUMBER', 'Undrafted']

    # season split
    train_seasons = ["2016/17", "2017/18", "2018/19", "2019/20"]
    test_seasons = ["2020/21"]

    # bool array for train and test set
    train = np.asarray(df_model['season'].isin(train_seasons))
    test = np.asarray(df_model['season'].isin(test_seasons))

    # get X and y
    X = df_model[features].values
    y = df_model['log_salary'].values
    df = df_model

    # train test split
    X_train, y_train, df_train = X[train], y[train], df[train]
    X_test, y_test, df_test = X[test], y[test], df[test]

    # remove nas
    bool_na = (y != 0) & (np.isfinite(X).all(axis=1)) & (~np.isnan(y))
    X = X[bool_na]
    y = y[bool_na].reshape(-1, 1)
    df = df[bool_na]

    # remove na rows train set
    bool_na_train = (y_train != 0) & (np.isfinite(X_train).all(axis=1)) & (~np.isnan(y_train))
    X_train = X_train[bool_na_train]
    y_train = y_train[bool_na_train].reshape(-1, 1)
    df_train = df_train[bool_na_train]

    # remove na rows test set
    bool_na_test = (y_test != 0) & (np.isfinite(X_test).all(axis=1)) & (~np.isnan(y_test))
    X_test = X_test[bool_na_test]
    y_test = y_test[bool_na_test].reshape(-1, 1)
    df_test = df_test[bool_na_test]

    return X_train, y_train, df_train, X_test, y_test, df_test, X, y, df

def wrapper_tune_fit(X_train, y_train, model, param_grid):
    """

    :param X_train:
    :param y_train:
    :param model:
    :param param_grid:
    :return:
    """

    # no tuning for OLS
    if param_grid == "OLS":
        return model.fit(X_train, y_train)

    # tune and return for rf and svr
    else:
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        # perform gridsearch
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        # tuned hyperparameters
        print(best_params)
        model.set_params(**best_params)

        print("Fit and score model with best params... \n")
        model.fit(X_train, y_train)

        return model

def select_model_grid(model_name: str="rf"):
    """

    :param model_name: rf, svr, ols
    :return: param_grid and model
    """

    if model_name == "rf":

        # rf
        # Create the parameter grid based on the results of random search
        param_grid = {
            'max_depth': [5, 7, 9, 12, 15],
            'min_samples_leaf': [2, 5, 7],
            'n_estimators': [75, 100, 125]
        }

        # Create a based model
        model = RandomForestRegressor()

    elif model_name=="svr":

        # svr
        # Create the parameter grid based on the results of random search
        param_grid = {'svr__kernel': ['rbf', 'linear'],
                      'svr__C': [1e-2, 3e-2, 7e-2],
                      'svr__epsilon': [1e-10, 1e-7, 1e-5]}

        # Create a based model
        model = make_pipeline(StandardScaler(), SVR())


    else:
        param_grid = "OLS"
        model = LinearRegression()

    return model, param_grid

def score_model(X_test, y_test, model):
    """

    :param X_test:
    :param y_test:
    :param model:
    :return:
    """
    # fit and predict

    return model.score(X_test, y_test)

def fit_predict_full(X_train, y_train, X_test, model):
    """wrapper to fit and predict on full data set

    :param X:
    :param y:
    :param model:
    :return:
    """
    # fit and predict
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    return prediction, model

def create_plot_dataset(prediction, y, df):
    """

    :param prediction:
    :param y:
    :return:
    """
    # create plot df
    df_plot = df
    df_plot['log_Salary'] = y.flatten()
    df_plot['log_Predicted'] = prediction.flatten()
    df_plot['Salary'] = np.exp(df_plot['log_Salary'])
    df_plot['Predicted'] = np.exp(df_plot['log_Predicted'])
    df_plot['log_Difference'] = df_plot['log_Salary'] - df_plot['log_Predicted']
    df_plot['Difference'] = df_plot['Salary'] - df_plot['Predicted']

    df_plot.to_csv("./data/tmp/mincer_plot.csv")

    return df_plot

def plot_mincer(df_plot, logarithm: bool=False):
    """

    :param df_plot:
    :return:
    """
    if not logarithm:
        fig = go.Figure(data=[
            go.Scatter(
                x=df_plot["Predicted"],
                y=df_plot["Salary"],
                mode="markers",
                marker=dict(
                    colorscale='Viridis',
                    color=df_plot["Difference"],
                    colorbar={"title": "Difference"},
                    line={"color": "#444"},
                    reversescale=False,
                    sizeref=45,
                    sizemode="diameter",
                    opacity=0.8
                )
            )
        ])

    else:
        fig = go.Figure(data=[
                go.Scatter(
                    x=df_plot["log_Predicted"],
                    y=df_plot["log_Salary"],
                    mode="markers",
                    marker=dict(
                        colorscale='viridis',
                        color=df_plot["Difference"],
                        colorbar={"title": "Difference"},
                        line={"color": "#444"},
                        reversescale=False,
                        sizeref=45,
                        sizemode="diameter",
                        opacity=0.8
                )
                )])

    return fig