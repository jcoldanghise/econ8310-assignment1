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

# Generate x and y matrices
eqn = """trips ~ -1 + year + month + day + hour"""
y,x = pt.dmatrices(eqn, data=df_train)

# Initialize and fit the model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
modelFit = model.gridsearch(np.asarray(x), y)

# Create x_test variable for forecasting
x_test = pt.build_design_matrices([x.design_info], df_test)

# Generate prediction
pred = modelFit.predict(x_test[0])