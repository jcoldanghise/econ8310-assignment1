import pandas as pd
import numpy as np

#from statsmodels.tsa.api import ExponentialSmoothing

from pygam import LinearGAM, s, f
import patsy as pt

# Load train and test data
df_test = pd.read_csv('assignment_data_test.csv')
df_train = pd.read_csv('assignment_data_train.csv')

# Convert Timestamp column to datetime 
df_test['Timestamp'] = pd.to_datetime(df_test.Timestamp)
df_train['Timestamp'] = pd.to_datetime(df_train.Timestamp)

# Specify value to be predicited (trips) and date range index (Timestamp) for train data
trips = df_train['trips']
trips.index = df_train['Timestamp']
trips.index.freq = trips.index.inferred_freq

# Specify date range index (Timestamp) for test data
df_test.index = df_test['Timestamp']
df_test.index.freq  = df_test.index.inferred_freq

# Train and fit an Exponential Smoothing model with trend component 
# model = ExponentialSmoothing(trips, trend='add',initialization_method="estimated")
# modelFit = model.fit(optimized=True)

# # Generate prediction on the test set 
# pred = modelFit.predict(start=df_test.index[0], end=df_test.index[-1])

# Train and fit a Generalized Additive model 
# Generate x and y matrices
eqn = """trips ~ -1 + year + month + 
      day + hour"""
y,x = pt.dmatrices(eqn, data=df_train)

# Fit the model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
modelFit = model.gridsearch(np.asarray(x), y)

# Get specified model parameters from test set
x_forecast = pt.build_design_matrices([x.design_info], df_test)

# Forecast trips from the test data
pred = modelFit.predict(x_forecast[0])




# Try prophet 
