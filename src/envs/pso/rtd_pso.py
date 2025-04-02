import math
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model    

def compile_rtd_pso_subfunction(reference_trajectory_func_y,
                                allowable_error_x = 100,
                                allowable_error_vx = 40,
                                allowable_error_vy = 90,
                                max_x_error = 200,
                                max_alpha_deg = 10,
                                max_vx_error = 40,
                                max_vy_error = 90,
                                mach_end = 1.0,
                                alpha_reward_weight = 100,
                                x_reward_weight = 100,
                                vy_reward_weight = 100,
                                vx_reward_weight = 100): # i.e. mach goes from 0 to 1.0, but can stop between 1.0 and 1.1
    
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        xr, yr, vxr, vyr, m  = reference_trajectory_func_y(y)
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound
        
        if mass_propellant <= 0:
            return False
        elif mach_number > mach_end:
            if abs(x - xr) <= allowable_error_x and \
                abs(vx - vxr) <= allowable_error_vx and \
                abs(vy - vyr) <= allowable_error_vy:
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
        elif mach_number > mach_end + 0.1:
            return True, 2
        elif abs(x - xr) > max_x_error:
            return True, 3
        elif y < -10:
            return True, 4
        elif abs(alpha) > math.radians(max_alpha_deg):
            return True, 5
        elif abs(vx - vxr) > max_vx_error:
            return True, 6
        elif abs(vy - vyr) > max_vy_error:
            return True, 7
        else:
            return False, 0

    def reward_func_lambda(state, done, truncated):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        reward = 0

        # Get the reference trajectory
        xr, _, vxr, vyr, m = reference_trajectory_func_y(y)
        # Special errors
        if y < 0:
            return 0

        reward += math.exp(-4 * (vx - vxr)**2/max_vx_error**2) * vx_reward_weight
        reward += math.exp(-4 * (vy - vyr)**2/max_vy_error**2) * vy_reward_weight
        reward += math.exp(-4 * (x - xr)**2/max_x_error**2) * x_reward_weight
        reward += math.exp(-4*math.degrees(alpha)**2/max_alpha_deg**2) * alpha_reward_weight

        # Done function
        if done:
            print(f'Done at time: {time}')
            reward += 100000

        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda


def compile_rtd_pso(flight_stage = 'subsonic'):
    assert flight_stage in ['subsonic','supersonic']
    reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y()
    
    if flight_stage == 'subsonic':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_pso_subfunction(reference_trajectory_func_y,
                                                                                                  machs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                                                                  allowable_error_x = 200,
                                                                                                  allowable_error_vx = 20,
                                                                                                  allowable_error_vy = 50,
                                                                                                  max_x_error = 200,
                                                                                                  max_alpha_deg = 0.5,
                                                                                                  max_vx_error = 20,
                                                                                                  max_vy_error = 50,
                                                                                                  mach_end = 1.0,
                                                                                                  alpha_reward_weight = 100,
                                                                                                  x_reward_weight = 100,
                                                                                                  vy_reward_weight = 100,
                                                                                                  vx_reward_weight = 100)
    elif flight_stage == 'supersonic':
        xt, yt, vxt, vyt, mt = terminal_state
        density_t, atmospheric_pressure_t, speed_of_sound_t = endo_atmospheric_model(yt)
        speed_t = math.sqrt(vxt**2 + vyt**2)
        mach_number_t = speed_t / speed_of_sound_t

        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_pso_subfunction(reference_trajectory_func_y,
                                                                                                  allowable_error_x = 100,
                                                                                                  allowable_error_vx = 40,
                                                                                                  allowable_error_vy = 90,
                                                                                                  max_x_error = 200,
                                                                                                  max_alpha_deg = 10,
                                                                                                  max_vx_error = 40,
                                                                                                  max_vy_error = 90,
                                                                                                  mach_end = mach_number_t,
                                                                                                  alpha_reward_weight = 100,
                                                                                                  x_reward_weight = 100,
                                                                                                  vy_reward_weight = 100,
                                                                                                  vx_reward_weight = 100)
    else:
        raise ValueError(f'Invalid flight stage: {flight_stage}')

    return reward_func_lambda, truncated_func_lambda, done_func_lambda