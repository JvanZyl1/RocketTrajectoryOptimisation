import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def get_dt():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo.csv')
    # Extract time and state columns
    time = data['t[s]']
    dt_array = np.diff(time)
    dt = np.mean(dt_array)
    return dt


def reference_trajectory_lambda_func_y():
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
    
    terminal_state = states[-1]
    
    return interpolate_state, terminal_state


def calculate_flight_path_angles(vx_s, vy_s):
    flight_path_angle = np.arctan2(vx_s, vy_s)
    flight_path_angle_deg = np.rad2deg(flight_path_angle)
    return flight_path_angle_deg