import math
import numpy as np
from src.TrajectoryGeneration.Transformations import calculate_flight_path_angles
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
    
def reward_func(state, done, truncated, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    reward = 0

    # Get the reference trajectory
    xr, _, vxr, vyr, m = reference_trajectory_func(y)
    gamma_r = calculate_flight_path_angles(vxr, vyr)

    # Special errors
    if y < 0:
        return 0

    alpha_reward_weight = 100
    x_reward_weight = 100
    vy_reward_weight = 100
    gamma_reward_weight = 100

    max_alpha_deg = 20
    max_x_error = 200
    max_vy_error = 40
    max_gamma_error = 4

    reward += math.exp(-4 * (gamma - gamma_r)**2/max_gamma_error**2) * gamma_reward_weight
    reward += math.exp(-4 * (vy - vyr)**2/max_vy_error**2) * vy_reward_weight
    reward += math.exp(-4 * (x - xr)**2/max_x_error**2) * x_reward_weight
    reward += math.exp(-4*math.degrees(alpha)**2/max_alpha_deg**2) * alpha_reward_weight

    # Done function
    if done:
        print(f'Done at time: {time}')
        reward += 5000

    return reward

def truncated_func(state, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Get the reference trajectory
    xr, yr, vxr, vyr, m  = reference_trajectory_func(y)

    # Errors
    error_x = abs(x - xr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)
    # Flight path angle (deg)
    gamma_r = calculate_flight_path_angles(vxr, vyr)
    gamma = calculate_flight_path_angles(vx, vy)
    error_gamma = abs(gamma - gamma_r)

    # If mass is depleted, return True
    if mass_propellant <= 0:
        return True
    elif error_x > 200:
        return True
    elif time > 10 and error_gamma > 20:
        return True
    elif y < -10:
        return True
    elif abs(alpha) > math.radians(10):
        return True
    elif error_vx > 40:
        return True
    elif error_vy > 90:
        return True
    else:
        return False


def done_func(state,
              terminal_state,
              allowable_error_x = 100,                    # [m]
              allowable_error_y = 250,                    # [m]
              allowable_error_flight_path_angle = 4):     # [deg]
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    xr, yr, vxr, vyr, m = terminal_state
    gamma_terminal = calculate_flight_path_angles(vxr, vyr)

    # Check if mass is depleted : should be truncated
    if mass_propellant >= 0 and \
        abs(x - xr) <= allowable_error_x and \
        abs(y - yr) <= allowable_error_y and \
        abs(math.degrees(gamma) - gamma_terminal) <= allowable_error_flight_path_angle and \
            abs(alpha) <= math.radians(5):
            return True
    else:
        return False
        

def compile_rtd_pso():
    reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y()
    reward_func_lambda = lambda state, done, truncated : reward_func(state,
                                                                     done,
                                                                     truncated,
                                                                     reference_trajectory_func_y)
    truncated_func_lambda = lambda state : truncated_func(state,
                                                                   reference_trajectory_func_y)
    
    done_func_lambda = lambda state : done_func(state,
                                                        terminal_state,
                                                        allowable_error_x = 100,
                                                        allowable_error_y = 100,
                                                        allowable_error_flight_path_angle = 2)
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda