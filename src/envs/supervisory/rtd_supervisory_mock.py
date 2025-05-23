import math
import pandas as pd

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model    

def compile_rtd_supervisory_test(flight_phase = 'subsonic'):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn']
    if flight_phase == 'subsonic':
        terminal_mach = 1.1
    elif flight_phase == 'supersonic':
        reference_data = pd.read_csv(f'data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')
        terminal_altitude = reference_data['y[m]'].iloc[-1]
    
    flip_over_boostbackburn_terminal_vx = -60
    dynamic_pressure_threshold = 10000

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if flight_phase in ['subsonic']:
            mach_number = speed / speed_of_sound
            if mach_number > terminal_mach:
                return True
            else:
                return False
        elif flight_phase == 'supersonic':
            if y > terminal_altitude:
                return True
            else:
                return False
        elif flight_phase == 'flip_over_boostbackburn':
            if vx < flip_over_boostbackburn_terminal_vx:
                return True
            else:
                return False
        elif flight_phase == 'ballistic_arc_descent':
            if dynamic_pressure > dynamic_pressure_threshold and \
                abs_alpha_effective < math.radians(3):
                return True
            else:
                return False
        elif flight_phase == 'landing_burn':
            if y < 1:
                return True
            else:
                return False
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)

        if flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']:
            if mass_propellant <= 0:
                return True, 1
            else:
                return False, 0
        elif flight_phase == 'ballistic_arc_descent':
            if dynamic_pressure > dynamic_pressure_threshold and \
                abs_alpha_effective > math.radians(3):
                return True, 1
            else:
                return False, 0
        elif flight_phase == 'landing_burn':
            if y < 1:
                return True, 1
            elif dynamic_pressure > 35000:
                return True, 2
            elif mass_propellant <= 0:
                return True, 3
            else:
                return False, 0
    
    def reward_func_lambda(state, done, truncated):
        return 0
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda