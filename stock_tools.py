import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import requests

def get_data(start_date,end_date=datetime.now(),interest='SPY',from_file=False):
    """
    :param start_date: datetime of when you want the interest to finish
    :param end_date: datetime of when you want the interest to finish
    :param interest: string what stock symbol are we interested in
    :param from_file: bool describing if data should be from file or internet

    :return: pandas DataFrame with stock information from Yahoo finance 

    This gets the current or previous data from yahoo finance for a symbol
    """
    raw_data = None
    if from_file:
        try:
            raw_data = pd.read_csv('%s.csv' % interest,index_col=0, parse_dates=True)
        except (IOError, NameError):
            try:
                print('Can not load from file. Getting from the internet')
                raw_data = pdr.get_data_yahoo(symbols=interest, start=start_date, end=end_date)
            except requests.exceptions.ConnectionError:
                print('You do not have an internet connection')
            finally:
                print('Can not obtain datafile.')
    else:
        try:
            raw_data = pdr.get_data_yahoo(symbols=interest, start=start_date, end=end_date)
        except requests.exceptions.ConnectionError:
            print('You do not have an internet connection')
        try:
            raw_data.to_csv('%s.csv' % interest, sep=',')
        except (IOError, NameError):
            print('Can not save to file.')
    return raw_data

def ohlc_adj(dat):
    """
    :param dat: pandas DataFrame containing the price information

    :return: pandas DataFrame which has been normalized

    This takes a DataFrame from yahoo finance and changes the Open, High and Low so that you get the adjusted values
    """
    return pd.DataFrame({"Open": dat["Open"] * dat["Adj Close"] / dat["Close"],
                       "High": dat["High"] * dat["Adj Close"] / dat["Close"],
                       "Low": dat["Low"] * dat["Adj Close"] / dat["Close"],
                       "Close": dat["Adj Close"],
                       "Volume": dat["Volume"]})

def _logic(x):
    """
    :param x: float which is to have the operation applied

    :return: -1 or 1 depending on the value of x

    This returns -1 or 1 when x is below or above 0
    """
    if x < 0:
        x = -1
    elif x > 0:
        x = 1
    return x

def calc_trades(dat,fast=20,slow=50):
    """
    :param dat: pandas DataFrame containing the price information
    :param fast: int describing the fast rolling mean period
    :param slow: int describing the slow rolling mean period

    :return: pandas DataFrame with the price, regime and Signals 

    This is a simple trading model which uses the rolling means to calculate when trades should occour
    """
    dat['fast'] = np.round(dat["Close"].rolling(window=fast, center=False).mean(), 2)
    dat['slow'] = np.round(dat["Close"].rolling(window=slow, center=False).mean(), 2)
    dat['diff'] = dat['fast'] - dat['slow']
    dat['regime'] = dat['diff'].map(_logic)
    temp = dat['regime'].ix[-1]
    dat['regime'].ix[-1] = 0
    dat['signal'] = np.sign(dat['regime'] - dat['regime'].shift(1))
    dat['regime'].ix[-1] = temp

    trades = pd.concat([
        pd.DataFrame({"Price": dat.loc[dat["signal"] == 1, "Open"],
                      "Regime": dat.loc[dat["signal"] == 1, 'regime'],
                      "Signal": "Buy"}),
        pd.DataFrame({"Price": dat.loc[dat["signal"] == -1, "Close"],
                      "Regime": dat.loc[dat["signal"] == -1, 'regime'],
                      "Signal": "Sell"})
    ])
    trades.sort_index(inplace=True)
    return trades

def fillin_low_periods(trades,dat, daily=False):
    """
    :param trades: pandas DataFrame with Buy and sell signals + the regime we're in 
    :param dat: pandas DataFrame containing the price information

    :return: pandas DataFrame with the lows in the periods filled in

    This function takes a pandas DataFrame with trades for buying and selling and fills in the low price and 
    creates a format which can be used in for backtesting 
    """
    if daily is False:
        long_profits = pd.DataFrame({"Price": [], "Profit": [], "End Date": []})
        for i in range(0,len(trades),2):
            long_profits = pd.concat([long_profits,
                                      pd.DataFrame({
                                          "Price": dat.ix[trades.index[i],"Open"],
                                          "Profit": dat.ix[trades.index[i+1]].Close - dat.ix[trades.index[i], "Open"],
                                          "End Date": trades.index[i+1],
                                          "Low": min(dat.ix[trades.index[i]:trades.index[i+1],"Low"])}, index=[trades.index[i]])])
        # long_profits = pd.DataFrame({
        #     "Price": trades.loc[(trades['Signal'] == 'Buy') & trades['Regime'] == 1, 'Price'],
        #     "Profit": pd.Series(trades["Price"] - trades["Price"].shift(1)).loc[
        #         trades.loc[(trades['Signal'].shift(1) == 'Buy') & (trades['Regime'].shift(1) == 1)].index].tolist(),
        #     "End Date": trades['Price'].loc[
        #         trades.loc[(trades['Signal'].shift(1) == 'Buy') & (trades['Regime'].shift(1) == 1)].index
        #     ].index
        # })
        long_profits.sort_index(inplace=True)
        #long_profits['Low'] = long_profits.apply(lambda row: min(dat.ix[row.name:row['End Date'], 'Low']), axis=1)
    else:
        long_profits = pd.DataFrame({"Price": [], "Profit": [], "End Date": []})
        for i in range(0,len(trades),2):
            long_profits = pd.concat([long_profits,
                                      pd.DataFrame({
                                          "Price": dat.ix[trades.index[i],"Open"],
                                          "Profit": dat.ix[trades.index[i]].Close - dat.ix[trades.index[i], "Open"],
                                          "End Date": trades.index[i],
                                          "Low": dat.ix[trades.index[i],"Low"]}, index=[trades.index[i]])])
        long_profits.sort_index(inplace=True)
    return  long_profits

def backtest(signals, cash, port_value = .1, batch = 100, stoploss = 0.2, commission=0.0025):
    """
    :param signals: pandas DataFrame containing buy and sell signals with stock prices and symbols, like that returned by ma_crossover_orders
    :param cash: integer for starting cash value
    :param port_value: maximum proportion of portfolio to risk on any single trade
    :param batch: Trading batch sizes

    :return: pandas DataFrame with backtesting results

    This function backtests strategies, with the signals generated by the strategies being passed in the signals DataFrame. A fictitious portfolio is simulated and the returns generated by this portfolio are reported.
    """
    results = pd.DataFrame({"Start Port. Value": [],
                                   "End Port. Value": [],
                                   "End Date": [],
                                   "Shares": [],
                                   "Share Price": [],
                                   "Trade Value": [],
                                   "Profit per Share": [],
                                   "Total Profit": [],
                                   "Commission": [],
                                   "Stop-Loss Triggered": []})
    for index, row in signals.iterrows():
        batches = np.floor(cash * port_value) // np.ceil(batch * row["Price"])  # Maximum number of batches of stocks invested in
        if batches <1:
            batches = 1

        trade_val = batches * batch * row["Price"]  # How much money is put on the line with each trade
        comm_buy = trade_val*commission
        cash =  cash - comm_buy
        if row["Low"] < (1 - stoploss) * row["Price"]:  # Account for the stop-loss
            share_profit = np.round((1 - stoploss) * row["Price"], 2)
            stop_trig = True
        else:
            share_profit = row["Profit"]
            stop_trig = False
        profit = share_profit * batches * batch  # Compute profits
        comm_sell = np.abs(profit*commission)
        cash = cash - comm_sell
        # Add a row to the backtest data frame containing the results of the trade
        results = results.append(pd.DataFrame({
            "Start Port. Value": cash + comm_buy+comm_sell,
            "End Port. Value": cash + profit,
            "End Date": row["End Date"],
            "Shares": batch * batches,
            "Share Price": row["Price"],
            "Trade Value": trade_val,
            "Profit per Share": share_profit,
            "Total Profit": profit,
            'Commission': comm_buy+comm_sell,
            "Stop-Loss Triggered": stop_trig
        }, index=[index]))
        cash = max(0, cash + profit)

    return results