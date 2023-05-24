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

# Separate Urban and Rural datasets
Urban_Syria = Syria[Syria['Urban'] == 'Urban'].dropna()
Rural_Syria = Syria[Syria['Urban'] == 'Rural'].dropna()

# Create the interaction term for the rural dataset
Rural_Syria['Drought_Albedo'] = Rural_Syria['Drought'] * Rural_Syria['Albedo']

# Create weights matrices for the urban and rural datasets
w_urban = weights.Queen.from_dataframe(Urban_Syria)
w_urban.transform = 'r'
w_rural = weights.Queen.from_dataframe(Rural_Syria)
w_rural.transform = 'r'

# Create the models
X_urban = Urban_Syria[['Drought', 'Albedo', 'Year', 'Month', 'X', 'Y']]
y_urban = Urban_Syria['NL']
X_rural = Rural_Syria[['Drought', 'Albedo', 'Year', 'Month', 'X', 'Y', 'Drought_Albedo']]
y_rural = Rural_Syria['NL']

# Store variable names
var_names_urban = ['CONSTANT'] + list(X_urban.columns) + ['W_dep_var']
var_names_rural = ['CONSTANT'] + list(X_rural.columns) + ['W_dep_var']

# Fit the spatial lag models
print("starting  urban_lag_model")
urban_lag_model = ML_Lag(y_urban.values.reshape((-1, 1)), X_urban.values, w_urban)
print("Finished  urban_lag_model")

print("Starting  rural_lag_model")
rural_lag_model = ML_Lag(y_rural.values.reshape((-1, 1)), X_rural.values, w_rural)
print("Finished  rural_lag_model")

# Fit the spatial error models
print("Starting  urban_error_model")
urban_error_model = ML_Error(y_urban.values.reshape((-1, 1)), X_urban.values, w_urban)
print("Finished  urban_error_model")

print("Starting  rural_error_model")
rural_error_model = ML_Error(y_rural.values.reshape((-1, 1)), X_rural.values, w_rural)
print("Finished  rural_error_model")

# Print the summaries and store
urban_lag_summary = str(urban_lag_model.summary)
rural_lag_summary = str(rural_lag_model.summary)
urban_error_summary = str(urban_error_model.summary)
rural_error_summary = str(rural_error_model.summary)


summaries = [urban_lag_summary, rural_lag_summary, urban_error_summary, rural_error_summary]
names = [var_names_urban, var_names_rural, var_names_urban, var_names_rural]

# Replace 'var_X' with actual variable names
for i, summary in enumerate(summaries):
    for j, name in enumerate(names[i]):
        summary = summary.replace(f'var_{j}', name)
    summaries[i] = summary


# Replace newline characters with LaTeX newline command
summaries = [summary.replace('\n', ' \\\\ ') for summary in summaries]

# Start the LaTeX document
latex_document = "\\documentclass{article} \n\\begin{document} \n"

# Add each summary to the LaTeX document
for summary in summaries:
    latex_document += "\\begin{verbatim} \n"
    latex_document += summary
    latex_document += "\n\\end{verbatim} \n"

# End the LaTeX document
latex_document += "\\end{document}"

# Write the LaTeX document to a .tex file
with open('model_summaries.tex', 'w') as f:
    f.write(latex_document)
