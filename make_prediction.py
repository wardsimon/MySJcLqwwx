from datetime import datetime
import Methods as models
import Predictors as predictors
import stock_tools as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import ML_predict as strat

# Create a template with the available variables
interest = 'SPY'
start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2010-12-31', '%Y-%m-%d')
initial_capital = 100000
cost = 0.025


# Get the data and correct for fluctuations
data = st.get_data(start_date,end_date,from_file=True)
corr_data = st.ohlc_adj(data)
# Create a predictors class which we will base our decisions from
pred = predictors.Predictors(corr_data)

# The data is far too noisy to make accurate predictions.
# We apply a 5 day exponential rolling filter. This should preserve
# shape and reduce noise.
pred.e_filter(5)

# This is where we calculate our benchmark.
# It is a rolling momentum strategy, where the moving averages,
# Stochastic Oscillator and MACD-Histogram are taken into account
# Note that this could be optimised by choosing parameters correctly.
pred_B = predictors.Predictors(corr_data)
pred_B.e_filter(5)
trades_raw = pred_B.movAVG_BuySell()
trades_raw.to_csv('basic_trades.csv', sep=',')
trades_period = st.fillin_low_periods(trades_raw,corr_data)
# Note that we DON'T use the smoothed data for backtesting.
results = st.backtest(trades_period,initial_capital,commission=0.0025)

# Just some fancy plotting
def highlight_trades(idx,ax,color='green'):
    i=0
    while i<len(idx):
        if idx[i] == idx[i+1]:
            ax.axvspan(idx[i]-pd.Timedelta(1, unit='d'), idx[i + 1]+pd.Timedelta(1, unit='d'), facecolor=color, edgecolor='none', alpha=.2)
        else:
            ax.axvspan(idx[i], idx[i+1], facecolor=color, edgecolor='none', alpha=.2)
        i+=2

# Plot our benchmark
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
corr_data.Close.plot(label="SPY", legend=True)
ax1.set_ylabel("Share Price [$]")
ax2 = plt.subplot(gs[1],sharex=ax1)
results["End Port. Value"].div(initial_capital).plot(label="Return", legend=True)
ax2.set_xlabel("Time")
ax2.set_ylabel("Cash Normalized")
highlight_trades(trades_raw.index,ax1)
highlight_trades(trades_raw.index,ax2)
plt.title("Basic momentum flow")
plt.savefig("Benchmark_return.png")

corr_new = corr_data.copy()
# Can we make extra profit by predicting the future?
corr_new["Avail"] = 0
corr_new.ix[corr_data.index[0]:trades_raw.index[0],"Avail"] = 1
for i in range(1,len(trades_raw)-1,2):
    corr_new.ix[trades_raw.index[i]:trades_raw.index[i + 1],"Avail"] = 1


# We now have a dataset where trades can be made.
pred_R = corr_new[corr_new.Avail == 1]
# This is the difference in daily percentage which we will look at
logic = (1 - pred.data.Close.shift(1).div(pred.data.Close))

# Start the simulation loop
days_previous = 252 # This is the number of previous days to study
trades = pd.DataFrame({"Price": [],"Regime": [], "Signal": []}) # Empty dataframe to put trades.
# Cycle through each day and simulate
for i in range(len(pred_R)):
    # Check if we have enough previous data
    if pred_R.index[i] > pred.props.index[days_previous+1]:
        # Do the simulation. See ML_BuySell.py for details.
        bs = strat.ML_BuySell(pred.props, pred_R.index[i], {'meanfractal', 'RStok0', 'mom', 'MACD_l'}, logic)
        if bs is not None:
            trades = pd.concat([trades,bs])

trades.to_csv('ml_trades_gap.csv', sep=',')
# trades = pd.read_csv('ml_trades_gap.csv',index_col=0, parse_dates=True)

# Fill in the low periods.
trades_period2 = st.fillin_low_periods(trades,corr_data,daily=True)
# Note that we DON'T use the smoothed data for backtesting.
results2 = st.backtest(trades_period2,initial_capital,commission=0.0025)

# Plot our stratergy.
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
corr_data.Close.plot(label="SPY", legend=True)
ax1.set_ylabel("Share Price [$]")
ax2 = plt.subplot(gs[1],sharex=ax1)
results2["End Port. Value"].div(initial_capital).plot(label="Return", legend=True)
ax2.set_xlabel("Time")
ax2.set_ylabel("Cash Normalized")
highlight_trades(trades.index,ax1,'r')
highlight_trades(trades.index,ax2,'r')
plt.title("Random Forest Trades")
plt.savefig("RF_return.png")

acc = (results2["Total Profit"]<0).sum()/(results2["Total Profit"]>0).sum()

trades_tot = pd.concat([trades_raw,trades])
trades_period3 = st.fillin_low_periods(trades_tot,corr_data)
Results_total = st.backtest(trades_period3,initial_capital,commission=0.0025)

# And now both together
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
corr_data.Close.plot(label="SPY", legend=True)
ax1.set_ylabel("Share Price [$]")
ax2 = plt.subplot(gs[1],sharex=ax1)
Results_total["End Port. Value"].div(initial_capital).plot(label="Return", legend=True)
ax2.set_xlabel("Time")
ax2.set_ylabel("Cash Normalized")
highlight_trades(trades_raw.index,ax1)
highlight_trades(trades_raw.index,ax2)
highlight_trades(trades.index,ax1,'r')
highlight_trades(trades.index,ax2,'r')
plt.title("Random Forest + Momentum Trades")
plt.savefig("RF+M_return.png")
