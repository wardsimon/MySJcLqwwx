from tensorflow.contrib import learn
from lstm import lstm_model, rnn_data
import pandas as pd
import numpy as np


def DL_BuySell(price, days_future, pred,limit=0.0051, days_previous=504, train_split=0.9, periodM = 9,
               fastM = 12, slowM = 26, upperS = 80, lowerS = 20, daysS = 3, periodS = 14):
    """
    This function takes the previous stock price and makes a LSTM deep learning network based on it to predict 
    the future stock price. It is bound by a percentage change limit, stokementric oscillator and momentum considerations.

    :param price: pandas DataFrame, All technical details. generated from Predictors class
    :param days_future: pandas DateTime, The timestamps you want to predict 
    :param limit: float, the minimum limit for which trades can occur 
    :param days_previous: int, How many previous days should be simulated
    :param train_split: float, Training/Testing split between (0, 1)
    :param fastM: Period to calculate the fast mean (Exponential)
    :param periodM: MACD Smoothing number of days
    :param slowM: Period to calculate the slow mean (Exponential)
    :param upperS: Upper bound of the stokementric oscillator
    :param lowerS: Lower bound of the stokementric oscillator
    :param daysS: Period to calculate the mean (stokementric oscillator)
    :param periodS: Number of days to smooth the stokementric oscillator
    :return: pandas DataFrame containing Buy and Sell commands.
    """

    LOG_DIR = '.opt_logs/lstm_stock'
    TIMESTEPS = 20
    RNN_LAYERS = [{'num_units': 5}]
    DENSE_LAYERS = [10, 10]
    TRAINING_STEPS = 100000
    BATCH_SIZE = 100
    PRINT_STEPS = TRAINING_STEPS / 100

    # Split into testing and training data.
    data = price.ix[price.index < days_future[0]]
    data = data.ix[-1*(days_previous):]

    nval = int(round(len(data) * (1 - train_split)))
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:], price.ix[days_future]

    labels = False
    train_x, val_x, test_x = (rnn_data(df_train, TIMESTEPS, labels=labels),
            rnn_data(df_val, TIMESTEPS, labels=labels),
            rnn_data(df_test, TIMESTEPS, labels=labels))
    labels = True
    train_y, val_y, test_y = (rnn_data(df_train, TIMESTEPS, labels=labels),
            rnn_data(df_val, TIMESTEPS, labels=labels),
            rnn_data(df_test, TIMESTEPS, labels=labels))
    X, y = dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

    # Train the model
    regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS,
                                                    learning_rate=0.05, optimizer="Adagrad"),
                                model_dir=LOG_DIR)

    # create a lstm instance and validation monitor
    validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                          every_n_steps=PRINT_STEPS,
                                                          early_stopping_rounds=1000)
    # Fit the data to the models
    regressor.fit(X['train'], y['train'],
                  monitors=[validation_monitor],
                  batch_size=BATCH_SIZE,
                  steps=TRAINING_STEPS)

    # Make the prediction
    predicted = list(regressor.predict(X['test']))

    # Can we make money?
    if (np.max(predicted)-np.min(predicted))/np.max(predicted) < limit:
        return


    index = price.ix[data.index[0]:days_future.index[-1]].index
    pred_val = np.concatenate((price.ix[data.index[0]:days_future.index[0]].values,predicted))
    C = pd.DataFrame({"Close": pred_val}, index=index)
    stok_Sig = pred.stokO_signal(upper=upperS, lower=lowerS, days=daysS, period=periodS, C=C)
    MACD_Sig = pred.MACD_signal_line(period=periodM, hist=True, signal=True, fast=fastM, slow=slowM, C=C)

    # Where do all the signals overlap
    C["regime"] = (stok_Sig + MACD_Sig)
    C["regime"].ix[C["regime"] == 2] = 1
    C["regime"].ix[C["regime"] == 1] = 0
    C["regime"].ix[C["regime"] == 0] = 0
    C["regime"].ix[C["regime"] == -1] = 0
    C["regime"].ix[C["regime"] == -2] = -1

    # We always have to end on a sell signal
    CC = C[C.regime != 0].copy()
    temp = CC['regime'].ix[-1]
    CC['regime'].ix[-1] = 0
    CC['signal'] = np.sign(CC['regime'] - CC['regime'].shift(1))
    CC['regime'].ix[-1] = temp

    # Make a dataframe containing the buy and sell signals for back testing.
    moves = pd.concat([
        pd.DataFrame({"Price": CC.loc[CC["signal"] == 1, "Close"],
                      "Regime": CC.loc[CC["signal"] == 1, 'regime'],
                      "Signal": "Buy"}),
        pd.DataFrame({"Price": CC.loc[CC["signal"] == -1, "Close"],
                      "Regime": CC.loc[CC["signal"] == -1, 'regime'],
                      "Signal": "Sell"})
    ])
    moves.sort_index(inplace=True)
    if moves.ix[0].Regime == -1:
        moves = moves.ix[1:]

    return moves
