from datetime import datetime
import Methods as models
import Predictors as predictors
import stock_tools as st
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
sns.set(font_scale = 1.5)

# Create a template with the available variables
interest = 'SPY'
start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2010-12-31', '%Y-%m-%d')

# Get the data and correct for fluctuations
data = st.get_data(start_date,end_date,from_file=True)
corr_data = st.ohlc_adj(data)
# Create a predictors class which we will base our decisions from
pred = predictors.Predictors(corr_data)

# The data is far too noisy to make accurate predictions.
# We apply a 5 day exponential rolling filter. This should preserve
# shape and reduce noise.
pred.e_filter(5)

# Make testing and training data for feature extraction.
imp = pred.props
i = 252*2
ndays = 252*2
forward_look = 2
ind = int(np.round(ndays * 0.8))
temp = (pred.data.Close.shift(1).div(pred.data.Close) > 1)*1
X_TRAIN = imp.ix[(i - ndays):(i - ndays + ind)]
Y_TRAIN = temp.ix[imp.ix[(i - ndays+forward_look):(i - ndays + ind+forward_look)].index]
X_TEST = imp.ix[(i - ndays + ind):i]
Y_TEST = temp.ix[imp.ix[(i - ndays + ind+forward_look):(i+forward_look)].index]

cv_params = {'max_depth': [3,5,7,9,11], 'min_child_weight': [1,3,5,7,9]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                            cv_params,
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(X_TRAIN, Y_TRAIN)

print(optimized_GBM.grid_scores_)
me = [x[1] for x in optimized_GBM.grid_scores_]
best = optimized_GBM.grid_scores_[me.index(max(me))][0]

# We see that a max depth of 3 and min_child_weight of 7 is the best
# Now optimise the learning rate and subsample
cv_params = {'learning_rate': [0.1, 0.01, 0.005], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed': 0, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic', 'max_depth': best["max_depth"],
              'min_child_weight': best["min_child_weight"]}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                            cv_params,
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(X_TRAIN, Y_TRAIN)
print(optimized_GBM.grid_scores_)
me = [x[1] for x in optimized_GBM.grid_scores_]
best = {**best, **optimized_GBM.grid_scores_[me.index(max(me))][0]}

# We see that a subsample of 0.9 and learning rate of 0.005 is the best
xgdmat = xgb.DMatrix(X_TRAIN, Y_TRAIN)
# Create our DMatrix to make XGBoost more efficient as it does not play with sklearn
our_params = {'eta': best["learning_rate"], 'seed': 0, 'subsample': best["subsample"], 'colsample_bytree': 0.8,
              'objective': 'binary:logistic', 'max_depth': best["max_depth"],
              'min_child_weight': best["min_child_weight"]}
# Grid Search CV optimized settings
cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 4000, nfold = 10,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error
print(cv_xgb.tail(5))

our_params = {'eta': best["learning_rate"], 'seed': 0, 'subsample': best["subsample"], 'colsample_bytree': 0.8,
              'objective': 'binary:logistic', 'max_depth': best["max_depth"],
              'min_child_weight': best["min_child_weight"]}
final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)

# Now we check out the feature importance
xgb.plot_importance(final_gb)
# We find that Stok0, meanfractal, mom and MACD_I and dvol are important.
# Now try the test data
testdmat = xgb.DMatrix(X_TEST)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
print(y_pred)
predicted = y_pred
predicted[predicted > 0.5] = 1
predicted[predicted <= 0.5] = 0
X_TEST["REAL"] = Y_TEST
X_TEST["PRED"] = predicted
ret = accuracy_score(predicted, Y_TEST), 1-accuracy_score(predicted, Y_TEST)
print("Accuracy is %s" % ret[0])
# We have obtained 0.930693069307 accuracy


# From this lest just make a scatter plot
m = []
m.append(models.ML(pred.props))
m[0].pred.PHH = (pred.data.Close.shift(1).div(pred.data.Close) > 1)*1
ax = pred.props.ix[m[0].pred.PHH == 0].plot.scatter(x='meanfractal', y='Stok0', label='Group 1',color="b")
a2 = pred.props.ix[m[0].pred.PHH == 1].plot.scatter(x='meanfractal', y='Stok0', label='Group 2',color="r",ax=ax)
