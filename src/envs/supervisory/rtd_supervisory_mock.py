import math
import pandas as pd

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model    

def compile_rtd_supervisory_test(flight_phase = 'subsonic'):
    assert flight_phase in ['subsonic', 'supersonic']
    if flight_phase == 'subsonic':
        terminal_mach = 1.1
    elif flight_phase == 'supersonic':
        reference_data = pd.read_csv(f'data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')
        y_f = reference_data['y[m]'].iloc[-1]
        vx_f = reference_data['vx[m/s]'].iloc[-1]
        vy_f = reference_data['vy[m/s]'].iloc[-1]
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_f)
        speed = math.sqrt(vx_f**2 + vy_f**2)
        terminal_mach = speed / speed_of_sound
    else:
        raise ValueError(f'Invalid flight stage: {flight_phase}')

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        mach_number = speed / speed_of_sound
        if mach_number > terminal_mach:
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
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated):
        return 0
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda