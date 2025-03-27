import math
import numpy as np
from src.TrajectoryGeneration.Transformations import calculate_flight_path_angles
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
    
def reward_func(state, done, truncated, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    reward = 900

    # Get the reference trajectory
    xr, _, vxr, vyr, m = reference_trajectory_func(y)
    gamma_r = calculate_flight_path_angles(vxr, vyr)

    # Special errors
    if y < 0:
        return 0
    
    # are not None or Nan
    assert xr is not None and not np.isnan(xr)
    assert vxr is not None and not np.isnan(vxr)
    assert vyr is not None and not np.isnan(vyr)
    assert gamma_r is not None and not np.isnan(gamma_r)

    reward -= abs((x - xr)/xr)
    reward -= abs((vx - vxr)/vxr)
    reward -= abs((vy - vyr)/vyr)
    reward -= abs((theta - gamma_r)/gamma_r)
    reward -= abs((gamma - gamma_r)/gamma_r)
    reward += (10 - abs(math.degrees(alpha)))/10

    if y < 1000:
        reward -= 100
    # Done function
    if done:
        print(f'Done at time: {time}')
        reward += 50

    reward /= 1e6

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
    elif y > 20000:
        if error_vx > 40:
            return True
        elif error_vy > 40:
            return True
        else:
            return False
    elif y < 20000:
        if error_vx > 20:
            return True
        elif error_vy > 20:
            return True
        else:
            return False
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