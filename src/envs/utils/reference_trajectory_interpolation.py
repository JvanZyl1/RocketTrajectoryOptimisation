import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


def get_dt():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')
    # Extract time and state columns
    time = data['t[s]']
    dt_array = np.diff(time)
    dt = np.mean(dt_array)
    return dt

def fix_csv():
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


def reference_trajectory_lambda():
    # Read the csv data/reference_trajectory/reference_trajectory_endo.csv
    # Has format: t[s], x[m], y[m], vx[m/s], vy[m/s], mass[kg]
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')
    
    # Extract time and state columns
    times = data['t[s]']
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values
    
    # Create an interpolation function for each state variable
    interpolators = [interp1d(times, states[:, i], kind='linear', fill_value="extrapolate") for i in range(states.shape[1])]
    
    # Return a function that takes in a time and returns the state
    def interpolate_state(t):
        return np.array([interpolator(t) for interpolator in interpolators])
    
    # Extract final time
    final_time = times.iloc[-1]
    return interpolate_state, final_time

def reference_trajectory_lambda_func_y():
    # Read the csv data/reference_trajectory/reference_trajectory_endo.csv
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')
    
    # Extract y and state columns
    y_values = data['y[m]']
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values
    
    # Create an interpolation function for each state variable based on y
    interpolators = [interp1d(y_values, states[:, i], kind='linear', fill_value="extrapolate") for i in range(states.shape[1])]
    
    # Return a function that takes in a y value and returns the state
    def interpolate_state(y):
        result = np.array([interpolator(y) for interpolator in interpolators])
        if np.any(np.isnan(result)):
            print(f"NaN detected in interpolation result at y = {y}: {result}")
        return result
    
    # Extract terminal state
    terminal_state = states[-1]
    
    return interpolate_state, terminal_state