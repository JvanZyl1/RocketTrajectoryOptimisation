import csv
import pandas as pd

sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]

# Parameters
m_strc = float(sizing_results['Actual structural mass stage 2'])*1000
mass_prop_0 = float(sizing_results['Actual propellant mass stage 2'])*1000
mass_0 = m_strc + mass_prop_0
v_ex = float(sizing_results['Exhaust velocity stage 2'])
T_e = float(sizing_results['Thrust engine stage 2'])

# Extract initial state
data = pd.read_csv('data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')

# Extract initial state
t_0 = data['Time [s]'].iloc[0]
x_0 = data['x[m]'].iloc[0]
y_0 = data['y[m]'].iloc[0]
vx_0 = data['vx[m/s]'].iloc[0]
vy_0 = data['vy[m/s]'].iloc[0]
theta_0 = data['theta[rad]'].iloc[0]
theta_dot_0 = data['theta_dot[rad/s]'].iloc[0]
alpha_0 = data['alpha[rad]'].iloc[0]