import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math
import numpy as np
import csv

from src.envs.env_wrapper import EnvWrapper_Skeleton
from src.controls.TrajectoryGeneration.Transformations import calculate_flight_path_angles
from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent

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
    # If absolute angle of attack is greater than 20 degrees
    elif abs(alpha) > 20:
        return True
    else:
        return False


def done_func(physics_state,
              terminal_state,
              allowable_error_x = 100,                    # [m]
              allowable_error_y = 100,                    # [m]
              allowable_error_flight_path_angle = 2):     # [deg]
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = physics_state
    xr, yr, vxr, vyr, m = terminal_state
    gamma_terminal = calculate_flight_path_angles(vxr, vyr)

    # Check if mass is depleted : should be truncated
    if mass_propellant >= 0 and \
        abs(x - xr) <= allowable_error_x and \
        abs(y - yr) <= allowable_error_y and \
        abs(gamma - gamma_terminal) <= allowable_error_flight_path_angle:
        return True
    else:
        return False
        

def create_env_funcs():
    reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
    reward_func_lambda = lambda physics_state, done, truncated : reward_func(physics_state,
                                                                             done,
                                                                             truncated,
                                                                             reference_trajectory_func,
                                                                             final_reference_time)
    truncated_func_lambda = lambda physics_state : truncated_func(physics_state,
                                                                   reference_trajectory_func,
                                                                   final_reference_time)
    
    terminal_state = reference_trajectory_func(final_reference_time)
    done_func_lambda = lambda physics_state : done_func(physics_state,
                                                        terminal_state,
                                                        allowable_error_x = 100,
                                                        allowable_error_y = 100,
                                                        allowable_error_flight_path_angle = 2)
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda


class vertical_rising_wrapped_env(EnvWrapper_Skeleton):
    def __init__(self,
                 sizing_needed_bool: bool = False,
                 print_bool: bool = False):
        env = rocket_model_endo_ascent(action_dim = 1,
                                        sizing_needed_bool = sizing_needed_bool)
        x_max = 20
        y_max = 300
        vx_max = 10
        vy_max = 30
        theta_max = math.radians(100)
        theta_dot_max = math.radians(1)
        gamma_max = math.radians(100)
        alpha_max = math.radians(10)
        # Read sizing results
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        mass_max = float(sizing_results['Initial mass (subrocket 0)'])*1000,             # mass [kg]

        altitude_error_max = y_max
        target_altitude_max = y_max
        state_max = np.array([x_max, y_max, vx_max, vy_max, theta_max, theta_dot_max, gamma_max, alpha_max, mass_max, altitude_error_max, target_altitude_max])
        super().__init__(env, print_bool, state_max)