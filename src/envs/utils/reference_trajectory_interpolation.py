import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def get_dt():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/reference_trajectory_ascent_control.csv')
    time = data['t[s]']
    dt_array = np.diff(time)
    dt = np.mean(dt_array)
    return dt

def fix_csv():
    data = pd.read_csv('data/reference_trajectory/SizingSimulation/reference_trajectory_endo.csv')
    data.interpolate(method='linear', inplace=True)
    data.to_csv('data/reference_trajectory/SizingSimulation/reference_trajectory_endo_clean.csv', index=False)

def reference_trajectory_lambda_func_y():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/reference_trajectory_ascent_control.csv')
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