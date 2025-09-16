import pandas as pd
import numpy as np

#from statsmodels.tsa.api import ExponentialSmoothing

from pygam import PoissonGAM, s
import patsy as pt

from sklearn.model_selection import train_test_split

# Load train and test data
df_test = pd.read_csv('assignment_data_test.csv')
df_train = pd.read_csv('assignment_data_train.csv')

# Convert Timestamp column to datetime 
df_test['Timestamp'] = pd.to_datetime(df_test.Timestamp)
df_train['Timestamp'] = pd.to_datetime(df_train.Timestamp)

# Cyclical encodings for periodic features
df_train["hour_sin"]  = np.sin(2*np.pi*df_train["hour"]/24.0)
df_train["hour_cos"]  = np.cos(2*np.pi*df_train["hour"]/24.0)
df_train["month_sin"] = np.sin(2*np.pi*df_train["month"]/12.0)
df_train["month_cos"] = np.cos(2*np.pi*df_train["month"]/12.0)


# Cyclical encodings for periodic features
df_test["hour_sin"]  = np.sin(2*np.pi*df_test["hour"]/24.0)
df_test["hour_cos"]  = np.cos(2*np.pi*df_test["hour"]/24.0)
df_test["month_sin"] = np.sin(2*np.pi*df_test["month"]/12.0)
df_test["month_cos"] = np.cos(2*np.pi*df_test["month"]/12.0)


# You can keep simple trend terms too (often useful)
feature_cols = [
    "year",          # slow trend across years
    "day",           # within-month trend
    # raw hour/month if you want (optionally comment out)
    "hour",
    "month",
    # cyclical
    "hour_sin","hour_cos",
    "month_sin","month_cos",
]


# Handle NAs if any
X = df_train[feature_cols].copy()
for c in X.columns:
    X[c] = X[c].astype(float)
    X[c] = X[c].fillna(X[c].median())

y = df_train['trips'].values

# ---------------------------
# 3) Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=123
)

# One smooth term per column; you can adjust n_splines if you like.
terms = None
for i in range(X.shape[1]):
    term = s(i, n_splines=20)
    terms = term if terms is None else terms + term


model = PoissonGAM(terms)

lam_grid = np.logspace(-3, 3, 9)   # try a modest range
modelFit = model.gridsearch(X_train, y_train, lam=lam_grid, progress=False)

# Handle NAs if any
X_forecast = df_test[feature_cols].copy()
for c in X_forecast.columns:
    X_forecast[c] = X_forecast[c].astype(float)
    X_forecast[c] = X_forecast[c].fillna(X[c].median())
    
    
 pred = modelFit.predict_mu(X_forecast)

# # Specify value to be predicited (trips) and date range index (Timestamp) for train data
# trips = df_train['trips']
# trips.index = df_train['Timestamp']
# trips.index.freq = trips.index.inferred_freq

# # Specify date range index (Timestamp) for test data
# df_test.index = df_test['Timestamp']
# df_test.index.freq  = df_test.index.inferred_freq

# Train and fit an Exponential Smoothing model with trend component 
# model = ExponentialSmoothing(trips, trend='add',initialization_method="estimated")
# modelFit = model.fit(optimized=True)

# # Generate prediction on the test set 
# pred = modelFit.predict(start=df_test.index[0], end=df_test.index[-1])

# # Train and fit a Generalized Additive model 
# # Generate x and y matrices
# eqn = """trips ~ -1 + year + month + 
#       day + hour"""
# y,x = pt.dmatrices(eqn, data=df_train)

# # Fit the model
# model = LinearGAM(s(0) + s(1) + s(2) + s(3))
# modelFit = model.gridsearch(np.asarray(x), y)

# # Get specified model parameters from test set
# x_forecast = pt.build_design_matrices([x.design_info], df_test)

# # Forecast trips from the test data
# pred = modelFit.predict(x_forecast[0])




# Try prophet 
