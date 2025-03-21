from src.envs.env_endo.init_vertical_rising import reference_trajectory_lambda_func_y
import numpy as np

import pandas as pd

# Load the CSV file
data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo.csv')

# Check for missing values
print("Missing values before handling:")
print(data.isnull().sum())

# Fill missing values using interpolation
data.interpolate(method='linear', inplace=True)

# Alternatively, fill with mean
# data.fillna(data.mean(), inplace=True)

# Check for missing values after handling
print("Missing values after handling:")
print(data.isnull().sum())

# Save the cleaned data back to CSV
data.to_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv', index=False)

y = np.linspace(-100, 10000, 1000)

for val in y:
    reference_trajectory_func, terminal_state = reference_trajectory_lambda_func_y()
    state = reference_trajectory_func(val)