import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def reference_trajectory_lambda():
    # Read the csv data/reference_trajectory_endo.csv
    # Has format: t[s], x[m], y[m], vx[m/s], vy[m/s], m[kg]
    data = pd.read_csv('data/reference_trajectory_endo.csv')
    
    # Extract time and state columns
    time = data['t[s]']
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'm[kg]']].values
    
    # Create an interpolation function for each state variable
    interpolators = [interp1d(time, states[:, i], kind='linear', fill_value="extrapolate") for i in range(states.shape[1])]
    
    # Return a function that takes in a time and returns the state
    def interpolate_state(t):
        return np.array([interpolator(t) for interpolator in interpolators])
    
    # Extract final time
    final_time = time[-1]
    
    return interpolate_state, final_time
    
def reward_func(physics_state, done, truncated, reference_trajectory_func, final_reference_time):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = physics_state

    # Get the reference trajectory
    reference_state = reference_trajectory_func(time)
    xr, yr, vxr, vyr, m = reference_state

    # Calculate the reward
    max_error_x = 10
    max_error_y = 100
    max_error_vx = 1
    max_error_vy = 5

    error_x = abs(x - xr)
    error_y = abs(y - yr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)

    reward = -error_x/max_error_x - error_y/max_error_y - error_vx/max_error_vx - error_vy/max_error_vy

    # Special errors
    if y < 0:
        reward -= 1000

    # Truncated function
    if truncated:
        reward -= final_reference_time - time

    # Done function
    if done:
        reward += 1000

    return reward

def create_quick_reward_func():
    reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
    reward_func_lambda = lambda physics_state, done, truncated : reward_func(physics_state,
                                                                             done,
                                                                             truncated,
                                                                             reference_trajectory_func,
                                                                             final_reference_time)
    return reward_func_lambda

def truncated_func(physics_state, reference_trajectory_func, final_reference_time):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = physics_state

    # Get the reference trajectory
    reference_state = reference_trajectory_func(time)
    xr, yr, vxr, vyr, m = reference_state

    # Errors
    error_x = abs(x - xr)
    error_y = abs(y - yr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)

    # Now check if time is greater than final_reference_time + 10 seconds
    if time > final_reference_time + 10:
        return True
    # Now check if error_x is greater than 

def done_func():
    pass
