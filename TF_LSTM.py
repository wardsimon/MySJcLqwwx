from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from lstm import lstm_model, load_csvdata
from datetime import datetime
import Predictors as predictors
import stock_tools as st
import matplotlib.pyplot as plt
import pandas as pd


# This is a test of a LSTM memory deep learning model to predict the SPY
# It is written using tensorflow and is the process of being updated to tensrflow 1.0
# The memory modules and the number of nodes have not be tuned, so predictions can be made better.

# In the future more predictors should be introduced.

# Create a template with the available variables
interest = 'SPY'
start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2010-12-31', '%Y-%m-%d')

# Get the data and correct for fluctuations
data = st.get_data(start_date, end_date, from_file=True)
corr_data = st.ohlc_adj(data)
# Create a predictors class which we will base our decisions from
pred = predictors.Predictors(corr_data)

# The data is far too noisy to make accurate predictions.
# We apply a 5 day exponential rolling filter. This should preserve
# shape and reduce noise.
pred.e_filter(5)

# Setup the default vairables
LOG_DIR = '.opt_logs/lstm_stock'
TIMESTEPS = 20
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

# We want to predict the Closing price. Not too much data
close = pred.data.Close.ix[(len(pred.props)-252*2):]

# Split the data into test and train
X, y = load_csvdata(close, TIMESTEPS, seperate=False)


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

# Calculate the error
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)
# MSE: 0.097317

# plot the data
actual = close.ix[len(close)-len(predicted):]
predicted = pd.Series(predicted, index=actual.index)

comp = pd.DataFrame({"Actual": actual,"Pred":predicted})

plot_predicted = pred.props.price.ix[len(pred.props.price)-len(predicted):].plot(label='SPY', legend=True)
plot_test = comp.Pred.plot(label='Predicted SPY', legend=True)
plt.ylabel("Share Price [$]")
plt.savefig("LSTM_prediction2.png")
plt.show()