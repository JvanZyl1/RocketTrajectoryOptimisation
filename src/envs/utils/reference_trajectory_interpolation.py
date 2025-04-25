import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def reference_trajectory_lambda_func_y(flight_phase):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
    if flight_phase in ['subsonic', 'supersonic']:
        data = pd.read_csv('data/reference_trajectory/ascent_controls/reference_trajectory_ascent_control.csv')
    elif flight_phase == 'flip_over_boostbackburn':
        data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/reference_trajectory_flip_over_and_boostbackburn_control.csv')
    elif flight_phase == 'ballistic_arc_descent':
        data = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/reference_trajectory_ballistic_arc_descent_control.csv')
    elif flight_phase == 're_entry_burn':
        data = pd.read_csv('data/reference_trajectory/re_entry_burn_controls/reference_trajectory_re_entry_burn_control.csv')
    y_values = data['y[m]']
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values
    interpolators = [interp1d(y_values, states[:, i], kind='linear', fill_value="extrapolate") for i in range(states.shape[1])]
    
    def interpolate_state(y):
        result = np.array([interpolator(y) for interpolator in interpolators])
        if np.any(np.isnan(result)):
            print(f"NaN detected in interpolation result at y = {y}: {result}")
        return result
    
    terminal_state = states[-1]
    return interpolate_state, terminal_state