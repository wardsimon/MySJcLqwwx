from datetime import datetime
import Methods as models
import Predictors as predictors
import stock_tools as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


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
trades_period = st.fillin_low_periods(trades_raw,corr_data)
# Note that we DON'T use the smoothed data for backtesting.
results = st.backtest(trades_period,initial_capital)

# Just some fancy plotting
def highlight_trades(idx,ax):
    i=0
    while i<len(idx):
         ax.axvspan(idx[i], idx[i+1], facecolor='green', edgecolor='none', alpha=.2)
         i+=2

# Plot our benchmark
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
corr_data.Close.plot()
ax1.set_ylabel("SPX [$]")
ax2 = plt.subplot(gs[1],sharex=ax1)
results["End Port. Value"].div(initial_capital).plot()
ax2.set_xlabel("Time")
ax2.set_ylabel("Cash Normalized")
highlight_trades(trades_raw.index,ax1)
highlight_trades(trades_raw.index,ax2)


corr_new = corr_data.copy()
# Can we make extra profit by predicting the future?
corr_new["Avail"] = 0
corr_new.ix[corr_data.index[0]:trades_raw.index[0],"Avail"] = 1
for i in range(1,len(trades_raw)-1,2):
    corr_new.ix[trades_raw.index[i]:trades_raw.index[i + 1],"Avail"] = 1
# We now have a dataset where trades can be made.
pred_R = predictors.Predictors(corr_new)


sess = np.arange(-.01, .02,0.001)
j = 0
shown = []
(pred.data.Close.shift(1).div(pred.data.Close)).plot(color='b')
for ress in res:
    shown.append(ress[0])
    shown[j]["PERC"] = 1
    for i in range(1,len(ress)):
        def f(x):
            if x > 0.1:
                x = sess[i]
            else:
                x = 0
            return x
    shown[j]["PERC"] = shown[j]["PERC"] + ((ress[i-1].PRED > ress[i].PRED)*sess[i])
    shown[j]["PERC"].plot(color = "r", alpha = (1 - float(j)/len(ress)))

# temp = pred.make_percentage_higher(20,0)
# temp.plot()
# y_ = pd.Series()
# ndays = 252
# clf = neighbors.KNeighborsClassifier(n_neighbors=6)
# inp = pred.props.dropna()
# for i in range(ndays, len(pred.props.index)):
#     x = inp.ix[i - ndays:i, {'meanfractal', 'loc_vol'}]
#     y = inp.ix[i - ndays:i, 'PHH'] > 0.5
#     clf.fit(x, y)
#     y_ = y_.append(pd.Series(clf.predict(x)[-1]), ignore_index=True)
# model = clf
# print("Predicted model accuracy: " + str(model.score(inp.ix[ndays:,{'meanfractal', 'loc_vol'}], inp.ix[ndays:,'PHH'] > 0.5)))

fig, ax1 = plt.subplots()
pred.data.Close.div(pred.data.Close[0]).plot()
ax2 = ax1.twinx()
pred.calc_atr(20).plot(color='r')

pred.days = 12
pred.min_shift = 0.03

pred.make_bollinger(20,2)
fig, ax1 = plt.subplots()
pred.data.Close.div(pred.data.Close[0]).plot()
ax2 = ax1.twinx()
pred.data["20Bol_U"].plot(color='r')
pred.data["20Bol_L"].plot(color='r')


(Long, Short) = pred.calc_Chandelier()
fig, ax1 = plt.subplots()
pred.data.Close.div(pred.data.Close[0]).plot()
ax2 = ax1.twinx()
Long.plot(color='r')
Short.plot(color='r')

# print(pred.props.head())
# print(pred.props["switch"].value_counts())

m = models.ML(pred.props)
m.randomForset_learn(5, 15)

mm = models.DL(pred.props)
mm.do_all(252)

mm.create_dataset()
mm.nn_init()
mm.nn_learn()


model3 = []
model1 = []
acc1 = []
acc3 = []
for i in range(1,6):
    model3.append(models.ML(pred.props))
    model3[i-1].randomForset_learn(5, i)
    acc3.append(model3[i-1].accuracy)
    model1.append(models.ML(pred.props))
    try:
        model1[i-1].knn_learn(5,i)
        acc1.append(model1[i - 1].accuracy)
    except ValueError:
        acc1.append(0.)
        print("Model can not be evaluated at %s" % i)

plt.plot(range(1, 6), 'r--', acc1, range(1, 6), acc3, 'b--')
plt.ylabel('Accuracy')
plt.xlabel('Site Length')
plt.show()
