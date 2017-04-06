import pandas as pd
import numpy as np
from random import sample
from sklearn.ensemble import RandomForestClassifier


def ML_BuySell(all_data, predictDate, predictors, previous_results, limit=0.0051,
               limit_comp=np.arange(-.015, 0.02, 0.001), days_previous=252, train_split=0.8, n=3,acc_limit=0.75):
    """
    
    :param all_data: pandas DataFrame, All technical details. generated from Predictors class
    :param predictDate: pandas DateTime, The timestamp you want to look at
    :param predictors: array, containing the names of the technical indicators used.
    :param previous_results: pandas DataFrame, containing the daily percentage change  
    :param limit: float, the minimum limit for which trades can occur 
    :param limit_comp: numpy array, a list of percentages to check
    :param days_previous: int, How many previous days should be simulated
    :param train_split: float, Training/Testing split between (0, 1)
    :param n: int, number of random forrest classifiers
    :param acc_limit: float, specifies the minimum accuracy for a trade to take place.
    :return: pandas DataFrame containing Buy and Sell commands.
    """
    ALLX_DATA = all_data.ix[all_data.index < predictDate, predictors]
    if len(ALLX_DATA) < days_previous:
        return

    ALLY_DATA = previous_results.ix[all_data.index <= predictDate].shift(-1)
    ALLY_DATA = ALLY_DATA.drop(ALLY_DATA.index[-1])

    fluc_m = []
    X_TEST_B = ALLX_DATA[(-1 * days_previous):]
    Y_TEST_B = ALLY_DATA[(-1 * days_previous):]

    PREDICT_X = all_data.ix[all_data.index == predictDate, predictors]
    if PREDICT_X.empty:
        return
    pred_v = []
    acc = []
    for x in np.nditer(limit_comp):
        indices = sample(range(days_previous), int(np.round(days_previous * train_split)))
        X_TRAIN = X_TEST_B.ix[indices]
        Y_TRAIN = Y_TEST_B.ix[indices]
        X_TEST = X_TEST_B.drop(X_TEST_B.index[indices])
        Y_TEST = Y_TEST_B.drop(Y_TEST_B.index[indices])
        fluc_m.append(RandomForestClassifier(n_estimators=n))
        fluc_m[-1].fit(X_TRAIN, 1*(Y_TRAIN > x))
        a = fluc_m[-1].score(X_TEST, 1*(Y_TEST > x))
        acc.append(a)
        pred_v.append(fluc_m[-1].predict(PREDICT_X)[0])

    change = 0
    for i in range(1, len(limit_comp)):
        l = (pred_v[i - 1] > pred_v[i])
        if l:
            change = change + (l* limit_comp[i])

    if change > limit:
        return pd.concat([
                pd.DataFrame({"Price": all_data.ix[all_data.index == predictDate, "price"],
                      "Regime": 1,
                      "Signal": "Buy"}),
                pd.DataFrame({"Price": all_data.ix[all_data.index == predictDate, "price"],
                      "Regime": -1,
                      "Signal": "Sell"})
            ])
    else:
        return None
