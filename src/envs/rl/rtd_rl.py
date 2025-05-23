import math
import pandas as pd
from scipy.interpolate import interp1d
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.load_initial_states import load_landing_burn_initial_state

def compile_rtd_rl_ascent(reference_trajectory_func_y,
                                learning_hyperparameters,
                                terminal_mach): # i.e. mach goes from 0 to 1.0, but can stop between 1.0 and 1.1
    machs = [hyperparameter[0] for hyperparameter in learning_hyperparameters]
    max_x_errors = [hyperparameter[1] for hyperparameter in learning_hyperparameters]
    max_vy_errors = [hyperparameter[2] for hyperparameter in learning_hyperparameters]
    max_vx_errors = [hyperparameter[3] for hyperparameter in learning_hyperparameters]
    max_alpha_degs = [hyperparameter[4] for hyperparameter in learning_hyperparameters]
    alpha_reward_weights = [hyperparameter[5] for hyperparameter in learning_hyperparameters]
    x_reward_weights = [hyperparameter[6] for hyperparameter in learning_hyperparameters]
    vy_reward_weights = [hyperparameter[7] for hyperparameter in learning_hyperparameters]
    vx_reward_weights = [hyperparameter[8] for hyperparameter in learning_hyperparameters]

    # Interpolate the learning hyperparameters
    f_max_x_error = interp1d(machs, max_x_errors, kind='linear', fill_value='extrapolate')
    f_max_vy_error = interp1d(machs, max_vy_errors, kind='linear', fill_value='extrapolate')
    f_max_vx_error = interp1d(machs, max_vx_errors, kind='linear', fill_value='extrapolate')
    f_max_alpha_deg = interp1d(machs, max_alpha_degs, kind='linear', fill_value='extrapolate')
    f_alpha_reward_weight = interp1d(machs, alpha_reward_weights, kind='linear', fill_value='extrapolate')
    f_x_reward_weight = interp1d(machs, x_reward_weights, kind='linear', fill_value='extrapolate')
    f_vy_reward_weight = interp1d(machs, vy_reward_weights, kind='linear', fill_value='extrapolate')
    f_vx_reward_weight = interp1d(machs, vx_reward_weights, kind='linear', fill_value='extrapolate')    
    
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return False
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0
        
        if mass_propellant >= 0 and mach_number > terminal_mach:
            return True
        else:
            return False
        
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return True, 0
        xr, yr, vxr, vyr, m  = reference_trajectory_func_y(y)

        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0

        # If mass is depleted, return True
        if mass_propellant <= 0:
            return True, 1
        elif mach_number > terminal_mach + 0.09:
            return True, 2
        elif abs(x - xr) > f_max_x_error(mach_number):
            return True, 3
        elif y < 0:
            return True, 4
        elif abs(alpha) > math.radians(f_max_alpha_deg(mach_number)):
            return True, 5
        elif abs(vx - vxr) > f_max_vx_error(mach_number):
            return True, 6
        elif abs(vy - vyr) > f_max_vy_error(mach_number):
            return True, 7
        else:
            return False, 0

    def reward_func_lambda(state, done, truncated, actions):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return 0
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0
        reward = 0

        # Get the reference trajectory
        xr, _, vxr, vyr, m = reference_trajectory_func_y(y)
        # Special errors
        if y < 0:
            return 0

        reward += math.exp(-4 * (vx - vxr)**2/f_max_vx_error(mach_number)**2) * f_vx_reward_weight(mach_number)
        reward += math.exp(-4 * (vy - vyr)**2/f_max_vy_error(mach_number)**2) * f_vy_reward_weight(mach_number)
        reward += math.exp(-4 * (x - xr)**2/f_max_x_error(mach_number)**2) * f_x_reward_weight(mach_number)
        reward += math.exp(-4*math.degrees(alpha)**2/f_max_alpha_deg(mach_number)**2) * f_alpha_reward_weight(mach_number)

        # Done function
        if done:
            reward += 2.5

        reward /= 10**4
        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_test_boostback_burn(theta_abs_error_max):
    flip_over_boostbackburn_terminal_vx = -20
    data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
    theta = data['theta[rad]'].values
    vy_vals = data['vy[m/s]'].values
    f_theta = interp1d(vy_vals, theta, kind='linear', fill_value='extrapolate')
    def theta_abs_error(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        theta_ref = f_theta(vy)
        return abs(theta_ref - theta)

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if vx < flip_over_boostbackburn_terminal_vx:
            return True
        else:
            return False
    
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

        if mass_propellant <= 0:
            return True, 1
        elif theta_abs_error(state) > theta_abs_error_max:
            return True, 2
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated, actions):
        reward = 1 - theta_abs_error(state)/theta_abs_error_max
        if done:
            reward += 0.25
        reward /= 100
        return reward
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_ballistic_arc_descent(dynamic_pressure_threshold = 10000):
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if dynamic_pressure > dynamic_pressure_threshold and \
            abs_alpha_effective < math.radians(3):
            return True
        else:
            return False
    
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if dynamic_pressure > dynamic_pressure_threshold - 2000 and \
            abs_alpha_effective > math.radians(5):
            return True, 1
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated, actions):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        abs_alpha_effective = abs(gamma - theta - math.pi)

        reward = (math.pi - abs_alpha_effective)/math.pi
        if done:
            reward += 3.5
        reward /= 100
        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = False):
    dynamic_pressure_threshold_training = 60000 # some ley-way for the landing burn
    max_alpha_effective = math.radians(20)
    y_0 = load_landing_burn_initial_state()[1]
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        if y > 0 and y < 5:
            if speed < 1:
                return True
            else:
                return False
        else:
            return False
    
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * air_density * speed**2
        if vy < 0:
            alpha_effective = abs(gamma - theta - math.pi)
        else:
            alpha_effective = abs(theta - gamma)
        if y < -10:
            print(f'Truncated state due to y < -10: x = {x}, y = {y}, vx = {vx}, vy = {vy}, theta = {theta}, gamma = {gamma}, alpha = {alpha}, mass = {mass}, mass_propellant = {mass_propellant}, time = {time}')
            return True, 1
        elif mass_propellant <= 0:
            return True, 2
        elif theta > math.pi + math.radians(2):
            return True, 3
        elif dynamic_pressure > 30000:
            return True, 4
        elif alpha_effective > 35000:
            return True, 5
        elif vy > 0.0:
            return True, 6
        else:
            return False, 0
    
    if not pure_throttle:
        def reward_func_lambda(state, done, truncated, actions):
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
            air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            speed = math.sqrt(vx**2 + vy**2)
            dynamic_pressure = 0.5 * air_density * speed**2
            alpha_effective = abs(gamma - theta - math.pi)
            reward = 0
            #reward += 1 - math.log(1 + alpha_effective)/(math.log(1+max_alpha_effective)) # [0, 1]
            if actions.ndim == 2:
                u0 = actions[0][0]
            else:
                u0 = actions[0]
            tau = (u0 + 1)/2
            reward += (1.5 - math.log(1 + alpha_effective)/(math.log(1+max_alpha_effective)) - tau*0.5)*(1-y/y_0) * 2/3
            # Throttle reward, u0 is normalised throttle (-1 to 1) so have to move to (0 to 1)
            
            #reward += (u0 + 1)/2*0.25
            #reward /= 1.25 # Scale reward to 1.0
            if y < 100: # Want to minimise the vy, vy = 30 -> r = 0.25
                reward += 1 - math.tanh((speed-15)/15)
            if truncated and y < 5:
                reward += 1 - math.tanh((speed-5)/5)
            if done: # Not looked at this yet
                reward += 5
            reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length) # n-step rewards scaling
            # CHANGED THIS REWARD FUNCTION
            return reward
    else:
        def reward_func_lambda(state, done, truncated, actions):
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
            air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            speed = math.sqrt(vx**2 + vy**2)
            dynamic_pressure = 0.5 * air_density * speed**2
            # positive terms
            if actions.ndim == 2:
                u0 = actions[0][0]
            else:
                u0 = actions[0]
            tau = (u0 + 1)/2
            reward = 0.0
            reward += (1 - tau)           # (0, 1)
            if y < 100.0:
                reward += 1 - math.tanh((speed - 15.0) / 15.0)   # (0, 2)
            if done:
                reward += 50.0
            # penalties
            penalty = 0.0
            if vy > 100.0:
                penalty += 2.0                                          # (0, 2)
            if dynamic_pressure > 25000:
                reward -= math.sqrt(31000 - dynamic_pressure)*2 # bit above to avoid numerical instability problems

            # offset provides non-negativity
            offset = 0.0
            reward = offset + reward - penalty
            reward /= 15            
            reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length) # n-step rewards scaling
            return reward
    return reward_func_lambda, truncated_func_lambda, done_func_lambda
        
        

def compile_rtd_rl(flight_phase, trajectory_length, discount_factor):
    assert flight_phase in ['subsonic','supersonic','flip_over_boostbackburn','ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle']
    if flight_phase not in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle']:
        reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y(flight_phase)

        # [[mach, max_x_error, max_vy_error, max_vx_error, max_alpha_deg, alpha_reward_weight, x_reward_weight, vy_reward_weight, vx_reward_weight], ...]
        subsonic_learning_hyperparameters = [
            # [mach,    max_x_error,    max_vy_error,   max_vx_error,   max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight,   vx_reward_weight]
            [0.0,     50,              10,               10,            0.5,              100,                    100,                100,                 100],
            [0.1,     50,              15,               10,             10,              100,                    100,                100,                 100],
            [0.2,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.3,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.4,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.5,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.6,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.7,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.8,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.9,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [1.0,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [1.1,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
        ]

        # For mach in range 1 to max mach append a mock config for now
        supersonic_learning_hyperparameters = [
            # [mach,    max_x_error,    max_vy_error,   max_vx_error,   max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight,   vx_reward_weight]
            [1.0,     100,            50,              9,              8,              100,                    100,                100,                  100],
            [1.1,     100,            60,             20,              8,              100,                    100,                100,                  100],
            [1.5,     100,            60,             20,              8,              100,                    100,                100,                  100],
            [1.75,    100,            60,             30,              8,              100,                    100,                100,                  100],
            [2.0,     100,            60,             40,              8,              100,                    100,                100,                  100],
            [2.25,    100,            60,             50,              8,              100,                    100,                100,                  100],
            [2.5,     100,            60,             60,              8,              100,                    100,                100,                  100],
            [2.75,    100,            60,             70,              8,              100,                    100,                100,                  100],
            [3.0,     100,            60,             80,              8,              100,                    100,                100,                  100],
            [3.25,    100,            60,             90,              8,              100,                    100,                100,                  100],
            [3.5,     100,            60,            100,              8,              100,                    100,                100,                  100],
            [3.75,    100,            60,            100,              8,              100,                    100,                100,                  100],
        ]
    
    if flight_phase == 'subsonic':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = subsonic_learning_hyperparameters,
                                                                                                  terminal_mach = 1.0)
    elif flight_phase == 'supersonic':
        # Extract maximum Mach Number
        xt, yt, vxt, vyt, mt = terminal_state
        print(f'Terminal state: {terminal_state}')
        density_t, atmospheric_pressure_t, speed_of_sound_t = endo_atmospheric_model(yt)
        speed_t = math.sqrt(vxt**2 + vyt**2)
        mach_number_t = speed_t / speed_of_sound_t
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = supersonic_learning_hyperparameters,
                                                                                                  terminal_mach = mach_number_t)
    elif flight_phase == 'flip_over_boostbackburn':
        theta_abs_error_max_rad = math.radians(4)
        reward_func_lambda, truncated_func_lambda, done_func_lambda =  compile_rtd_rl_test_boostback_burn(theta_abs_error_max_rad)
    elif flight_phase == 'ballistic_arc_descent':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ballistic_arc_descent(dynamic_pressure_threshold = 10000)
    elif flight_phase == 'landing_burn' or flight_phase == 'landing_burn_ACS':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = False)
    elif flight_phase == 'landing_burn_pure_throttle':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = True)
    else:
        raise ValueError(f'Invalid flight stage: {flight_phase}')

    return reward_func_lambda, truncated_func_lambda, done_func_lambda