import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pysal.model.spreg import ML_Lag, ML_Error
from pysal.lib import weights
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder

# Loading the data
Syria = gpd.read_file("merged_albedo_NL_Urban/syria_Merged_50.shp")

# Convert Urban to categorical
Syria['Urban'] = Syria['Urban'].apply(lambda x: 'Urban' if x == 1 else 'Rural')

# Create a label (category) encoder object
le = LabelEncoder()

# Fit the encoder to the pandas column
le.fit(Syria['Month'])

# Apply the fitted encoder to the pandas column
Syria['Month'] = le.transform(Syria['Month'])

# Create weights matrix for the full dataset
#w_full = weights.Queen.from_dataframe(Syria)
#w_full.transform = 'r'

# Separate Urban and Rural datasets
Urban_Syria = Syria[Syria['Urban'] == 'Urban'].dropna()
Rural_Syria = Syria[Syria['Urban'] == 'Rural'].dropna()

# Create weights matrices for the urban and rural datasets
w_urban = weights.Queen.from_dataframe(Urban_Syria)
w_urban.transform = 'r'
w_rural = weights.Queen.from_dataframe(Rural_Syria)
w_rural.transform = 'r'

# Create the models
X_urban = Urban_Syria[['Drought', 'Albedo', 'Year', 'Month', 'X', 'Y']]
y_urban = Urban_Syria['NL']
X_rural = Rural_Syria[['Drought', 'Albedo', 'Year', 'Month', 'X', 'Y']]
y_rural = Rural_Syria['NL']

# Fit the spatial lag models
print("starting  urban_lag_model")
urban_lag_model = ML_Lag(y_urban.values.reshape((-1, 1)), X_urban.values, w_urban)
print("Finished  urban_lag_model")

print("Starting  rural_lag_model")
#rural_lag_model = ML_Lag(y_rural.values.reshape((-1, 1)), X_rural.values, w_rural)
print("Finished  rural_lag_model")

# Fit the spatial error models
print("Starting  urban_error_model")
urban_error_model = ML_Error(y_urban.values.reshape((-1, 1)), X_urban.values, w_urban)
print("Finished  urban_error_model")

print("Starting  rural_error_model")
#rural_error_model = ML_Error(y_rural.values.reshape((-1, 1)), X_rural.values, w_rural)
print("Finished  rural_error_model")


# Print the summaries
print("Urban Lag Model Summary:\n", urban_lag_model.summary)
#print("Rural Lag Model Summary:\n", rural_lag_model.summary)
print("Urban Error Model Summary:\n", urban_error_model.summary)
#print("Rural Error Model Summary:\n", rural_error_model.summary)
