import math
import pandas as pd
from scipy.interpolate import interp1d
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model    

def compile_rtd_pso_ascent(reference_trajectory_func_y,
                                learning_hyperparameters,
                                terminal_mach): # i.e. mach goes from 0 to 1.0, but can stop between 1.0 and 1.1
    machs = [hyperparameter[0] for hyperparameter in learning_hyperparameters]
    max_x_errors = [hyperparameter[1] for hyperparameter in learning_hyperparameters]
    max_vy_errors = [hyperparameter[2] for hyperparameter in learning_hyperparameters]
    max_alpha_degs = [hyperparameter[3] for hyperparameter in learning_hyperparameters]
    alpha_reward_weights = [hyperparameter[4] for hyperparameter in learning_hyperparameters]
    x_reward_weights = [hyperparameter[5] for hyperparameter in learning_hyperparameters]
    vy_reward_weights = [hyperparameter[6] for hyperparameter in learning_hyperparameters]

    # Interpolate the learning hyperparameters
    f_max_x_error = interp1d(machs, max_x_errors, kind='linear', fill_value='extrapolate')
    f_max_vy_error = interp1d(machs, max_vy_errors, kind='linear', fill_value='extrapolate')
    f_max_alpha_deg = interp1d(machs, max_alpha_degs, kind='linear', fill_value='extrapolate')
    f_alpha_reward_weight = interp1d(machs, alpha_reward_weights, kind='linear', fill_value='extrapolate')
    f_x_reward_weight = interp1d(machs, x_reward_weights, kind='linear', fill_value='extrapolate')
    f_vy_reward_weight = interp1d(machs, vy_reward_weights, kind='linear', fill_value='extrapolate')
    
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound
        
        if mass_propellant >= 0 and mach_number > terminal_mach:
            return True
        else:
            return False
        
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        xr, yr, vxr, vyr, m  = reference_trajectory_func_y(y)

        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound

        # If mass is depleted, return True
        if mass_propellant <= 0:
            return True, 1
        elif mach_number > terminal_mach + 0.09:
            return True, 2
        elif abs(x - xr) > f_max_x_error(mach_number):
            return True, 3
        elif y < -10:
            return True, 4
        elif abs(alpha) > math.radians(f_max_alpha_deg(mach_number)):
            return True, 5
        # 6 was vx
        elif abs(vy - vyr) > f_max_vy_error(mach_number):
            return True, 7
        else:
            return False, 0

    def reward_func_lambda(state, done, truncated):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound
        reward = 0

        # Get the reference trajectory
        xr, _, vxr, vyr, m = reference_trajectory_func_y(y)
        # Special errors
        if y < 0:
            return 0

        reward += math.exp(-4 * (vy - vyr)**2/f_max_vy_error(mach_number)**2) * f_vy_reward_weight(mach_number)
        reward += math.exp(-4 * (x - xr)**2/f_max_x_error(mach_number)**2) * f_x_reward_weight(mach_number)
        reward += math.exp(-4*math.degrees(alpha)**2/f_max_alpha_deg(mach_number)**2) * f_alpha_reward_weight(mach_number)

        # Done function
        if done:
            print(f'Done at time: {time}')
            reward += 100000

        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_pso_test_boostback_burn(theta_abs_error_max):
    flip_over_boostbackburn_terminal_vx = -150
    data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
    theta = data['theta[rad]'].values
    y = data['y[m]'].values
    f_theta = interp1d(y, theta, kind='linear', fill_value='extrapolate')
    def theta_abs_error(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        theta_ref = f_theta(y)
        return abs(theta_ref - theta)

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if vx < flip_over_boostbackburn_terminal_vx:
            return True
        else:
            return False
    
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound

        if mass_propellant <= 0:
            return True, 1
        elif theta_abs_error(state) > theta_abs_error_max:
            return True, 2
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        reward = 1 - f_theta(y)/theta_abs_error_max
        if done:
            reward =+ 500
        return reward
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda


def compile_rtd_pso(flight_phase = 'subsonic'):
    assert flight_phase in ['subsonic','supersonic', 'flip_over_boostbackburn']
    reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y(flight_phase)

    # Extract maximum Mach Number
    xt, yt, vxt, vyt, mt = terminal_state
    density_t, atmospheric_pressure_t, speed_of_sound_t = endo_atmospheric_model(yt)
    speed_t = math.sqrt(vxt**2 + vyt**2)
    mach_number_t = speed_t / speed_of_sound_t

    # [[mach, max_x_error, max_vy_error, max_alpha_deg, alpha_reward_weight, x_reward_weight, vy_reward_weight], ...]
    subsonic_learning_hyperparameters = [
        # [mach,    max_x_error,    max_vy_error,      max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight]
          [0.0,     50,              4,                 4,              100,                    100,                100],
          [0.1,     50,              4,                 4,              100,                    100,                100],
          [0.2,     50,              4,                 4,              100,                    100,                100],
          [0.3,     50,              4,                 4,              100,                    100,                100],
          [0.4,     50,              4,                 4,              100,                    100,                100],
          [0.5,     50,              4,                 4,              100,                    100,                100],
          [0.6,     50,              4,                 4,              100,                    100,                100],
          [0.7,     50,              4,                 4,              100,                    100,                100],
          [0.8,     50,              5,                 4,              100,                    100,                100],
          [0.9,     50,              6,                 4,              100,                    100,                100],
          [1.0,     50,              7,                 4,              100,                    100,                100],
          [1.1,     50,              7,                 4,              100,                    100,                100],
    ]

    # For mach in range 1 to max mach append a mock config for now
    supersonic_learning_hyperparameters = [
        # [mach,    max_x_error,    max_vy_error,  max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight]
          [1.0,     100,            50,             2,              250,                    100,                100],
          [1.1,     100,            60,             2,              250,                    100,                100],
          [1.5,     100,            60,             2,              250,                    100,                100],
          [1.75,    100,            60,             2,              250,                    100,                100],
          [2.0,     100,            60,             2,              250,                    100,                100],
          [2.25,    100,            60,             2,              250,                    100,                100],
          [2.5,     100,            60,             2,              250,                    100,                100],
          [2.75,    100,            60,             2,              250,                    100,                100],
          [3.0,     100,            60,             2,              250,                    100,                100],
          [3.25,    100,            60,             2,              250,                    100,                100],
          [3.5,     100,            60,             2,              250,                    100,                100],
          [3.75,    100,            60,             2,              250,                    100,                100],
    ]
    
    if flight_phase == 'subsonic':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_pso_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = subsonic_learning_hyperparameters,
                                                                                                  terminal_mach = 1.0)
    elif flight_phase == 'supersonic':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_pso_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = supersonic_learning_hyperparameters,
                                                                                                  terminal_mach = mach_number_t)
    elif flight_phase == 'flip_over_boostbackburn':
        theta_abs_error_max_rad = math.radians(1)
        reward_func_lambda, truncated_func_lambda, done_func_lambda =  compile_rtd_pso_test_boostback_burn(theta_abs_error_max_rad)
    else:
        raise ValueError(f'Invalid flight stage: {flight_phase}')

    return reward_func_lambda, truncated_func_lambda, done_func_lambda