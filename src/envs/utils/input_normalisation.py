import math
import numpy as np
import pandas as pd

def subsonic_input_normalisation():
    file_path = 'data/reference_trajectory/ascent_controls/subsonic_state_action_ascent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values
    # Find max absolute values of each state
    x_norm_val = np.max(np.abs(states[:, 0])) + 100
    y_norm_val = np.max(np.abs(states[:, 1])) + 500
    vx_norm_val = np.max(np.abs(states[:, 2])) + 5
    vy_norm_val = np.max(np.abs(states[:, 3])) + 50
    theta_norm_val = np.max(np.abs(states[:, 4])) + math.radians(2)
    theta_dot_norm_val = np.max(np.abs(states[:, 5]))*2.5
    alpha_norm_val = np.max(np.abs(states[:, 6])) + math.radians(3)
    mass_norm_val = np.max(np.abs(states[:, 7]))

    # np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals

def supersonic_input_normalisation():
    file_path = 'data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values
    # Find max absolute values of each state
    x_norm_val = np.max(np.abs(states[:, 0])) + 2500
    y_norm_val = np.max(np.abs(states[:, 1])) + 5000
    vx_norm_val = np.max(np.abs(states[:, 2])) + 100
    vy_norm_val = np.max(np.abs(states[:, 3])) + 150
    theta_norm_val = np.max(np.abs(states[:, 4])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 5]))*2.5
    alpha_norm_val = np.max(np.abs(states[:, 6])) + math.radians(3)
    mass_norm_val = np.max(np.abs(states[:, 7]))

    # np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals

def flip_over_boostbackburn_input_normalisation():
    file_path = 'data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv'
    data = pd.read_csv(file_path)
    states = data[['theta[rad]', 'theta_dot[rad/s]']].values
    # Find max absolute values of each state
    theta_norm_val = np.max(np.abs(states[:, 0])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 1]))*2.5

    # np.array([theta, theta_dot])

    input_normalisation_vals = np.array([theta_norm_val, theta_dot_norm_val])

    return input_normalisation_vals

def ballistic_arc_descent_input_normalisation():
    file_path = 'data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'gamma[rad]']].values
    # Find max absolute values of each state
    theta_norm_val = np.max(np.abs(states[:, 0])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 1]))*2.5
    gamma_norm_val = np.max(np.abs(states[:, 3])) + math.radians(5)
    alpha_norm_val = np.max(np.abs(states[:, 2])) + math.radians(5)

    input_normalisation_vals = np.array([theta_norm_val, theta_dot_norm_val, gamma_norm_val, alpha_norm_val])

    return input_normalisation_vals
    
def re_entry_burn_input_normalisation():
    file_path = 'data/reference_trajectory/re_entry_burn_controls/state_action_re_entry_burn_control.csv'
    data = pd.read_csv(file_path)
    states = data[['y[m]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'gamma[rad]', 'alpha[rad]', 'mass[kg]', 'x[m]', 'vx[m/s]']].values
    # Find max absolute values of each state
    y_norm_val = np.max(np.abs(states[:, 0])) + 4000
    vy_norm_val = np.max(np.abs(states[:, 1])) + 100
    theta_norm_val = np.max(np.abs(states[:, 2])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 3]))*2.5
    gamma_norm_val = np.max(np.abs(states[:, 5])) + math.radians(5)
    alpha_norm_val = np.max(np.abs(states[:, 4])) + math.radians(5)
    mass_norm_val = np.max(np.abs(states[:, 6])) + 100
    x_norm_val = np.max(np.abs(states[:, 7])) + 2000
    vx_norm_val = np.max(np.abs(states[:, 8])) + 25

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, gamma_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals

def landing_burn_input_normalisation():
    #action_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
    file_path_ballistic_arc = 'data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv'
    data_ballistic_arc = pd.read_csv(file_path_ballistic_arc)
    states_ballistic_arc = data_ballistic_arc[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values
    x_norm_val = np.min(np.abs(states_ballistic_arc[:, 0])) + 5000
    y_norm_val = np.max(np.abs(states_ballistic_arc[:, 1])) + 100
    vx_norm_val = np.max(np.abs(states_ballistic_arc[:, 2]))
    vy_norm_val = np.max(np.abs(states_ballistic_arc[:, 3])) + 800
    theta_norm_val = math.pi * 3/2
    theta_dot_norm_val = 0.1
    alpha_norm_val = math.pi * 7/4
    mass_norm_val = np.max(np.abs(states_ballistic_arc[:, 7])) + 800

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals




def find_input_normalisation_vals(flight_phase : str):
    if flight_phase == 'subsonic':
        return subsonic_input_normalisation()
    elif flight_phase == 'supersonic':
        return supersonic_input_normalisation()
    elif flight_phase == 'flip_over_boostbackburn':
        return flip_over_boostbackburn_input_normalisation()
    elif flight_phase == 'ballistic_arc_descent':
        return ballistic_arc_descent_input_normalisation()
    elif flight_phase == 're_entry_burn':
        return re_entry_burn_input_normalisation()
    elif flight_phase == 'landing_burn':
        return landing_burn_input_normalisation()
    else:
        raise ValueError(f"Invalid flight phase: {flight_phase}")