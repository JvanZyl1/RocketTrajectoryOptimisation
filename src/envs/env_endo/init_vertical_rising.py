import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math
from src.controls.TrajectoryGeneration.Transformations import calculate_flight_path_angles


def get_dt():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo.csv')
    # Extract time and state columns
    time = data['t[s]']
    dt_array = np.diff(time)
    dt = np.mean(dt_array)
    return dt

def reference_trajectory_lambda():
    # Read the csv data/reference_trajectory/reference_trajectory_endo.csv
    # Has format: t[s], x[m], y[m], vx[m/s], vy[m/s], mass[kg]
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo.csv')
    
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
    
def reward_func(state, done, truncated, reference_trajectory_func, final_reference_time):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Get the reference trajectory
    reference_state = reference_trajectory_func(time)
    xr, yr, vxr, vyr, m = reference_state

    # Calculate the reward
    pos_error = math.sqrt((x - xr)**2 + (y - yr)**2)
    reward = pos_error/1000

    # Special errors
    if y < 0:
        reward -= 1000

    # Truncated function
    if truncated:
        reward -= (final_reference_time - time)*10

    # Done function
    if done:
        reward += 1000

    return reward

def truncated_func(state, reference_trajectory_func, final_reference_time):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Get the reference trajectory
    reference_state = reference_trajectory_func(time)
    xr, yr, vxr, vyr, m = reference_state

    # Errors
    error_x = abs(x - xr)
    error_y = abs(y - yr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)

    # If mass is depleted, return True
    if mass_propellant <= 0:
        return True
    # Now check if time is greater than final_reference_time + 10 seconds
    elif time > final_reference_time + 10:
        return True
    # Now check if error_x is greater than 1000m
    elif error_x > 1000:
        return True
    # Now check if error_y is greater than 1000m
    elif error_y > 1000:
        return True
    else:
        return False


def done_func(state,
              terminal_state,
              allowable_error_x = 100,                    # [m]
              allowable_error_y = 100,                    # [m]
              allowable_error_flight_path_angle = 2):     # [deg]
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    xr, yr, vxr, vyr, m = terminal_state
    gamma_terminal = calculate_flight_path_angles(vxr, vyr)

    # Check if mass is depleted : should be truncated
    if mass_propellant >= 0 and \
        abs(x - xr) <= allowable_error_x and \
        abs(y - yr) <= allowable_error_y and \
        abs(math.degrees(gamma) - gamma_terminal) <= allowable_error_flight_path_angle:
        return True
    else:
        return False
        

def create_env_funcs():
    reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
    reward_func_lambda = lambda state, done, truncated : reward_func(state,
                                                                     done,
                                                                     truncated,
                                                                     reference_trajectory_func,
                                                                     final_reference_time)
    truncated_func_lambda = lambda state : truncated_func(state,
                                                                   reference_trajectory_func,
                                                                   final_reference_time)
    
    terminal_state = reference_trajectory_func(final_reference_time)
    done_func_lambda = lambda state : done_func(state,
                                                        terminal_state,
                                                        allowable_error_x = 100,
                                                        allowable_error_y = 100,
                                                        allowable_error_flight_path_angle = 2)
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda