import csv
import pandas as pd


## Boostback burn parameters
sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]
n_e = int(sizing_results['Number of engines gimballed stage 1'])
thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1'])
v_ex = float(sizing_results['Exhaust velocity stage 1'])
T_bb = n_e * thrust_per_engine_no_losses
mdot_bb = T_bb / v_ex

# Initial conditions boostback
data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
# time[s],x[m],y[m],vx[m/s],vy[m/s],theta[rad],theta_dot[rad/s],alpha[rad],gamma[rad],mass[kg],mass_propellant[kg],gimbalanglecommanded[deg],u0

# find time = time[0] + 15 s (so flipped over)
initial_time = data['time[s]'].iloc[0]
target_time = initial_time + 15

# Find the closest time point to target_time
closest_idx = (data['time[s]'] - target_time).abs().idxmin()
boostback_0_time = data['time[s]'].iloc[closest_idx]
boostback_0_mass = data['mass[kg]'].iloc[closest_idx]
boostback_0_x = data['x[m]'].iloc[closest_idx]
boostback_0_vx = data['vx[m/s]'].iloc[closest_idx]

print(f"Boostback initial conditions at t = {boostback_0_time:.2f} s:")
print(f"Mass: {boostback_0_mass:.2f} kg")
print(f"Position x: {boostback_0_x:.2f} m")
print(f"Velocity vx: {boostback_0_vx:.2f} m/s")

# Ballistic arc terminal time
data = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
# final time
ballistic_2_time = data['time[s]'].iloc[-1]
ballistic_2_mass = data['mass[kg]'].iloc[-1]
ballistic_2_x = data['x[m]'].iloc[-1]
ballistic_2_vx = data['vx[m/s]'].iloc[-1]

print(f"Ballistic arc terminal time: {ballistic_2_time:.2f} s")
print(f"Ballistic arc terminal mass: {ballistic_2_mass:.2f} kg")
print(f"Ballistic arc terminal position x: {ballistic_2_x:.2f} m")
print(f"Ballistic arc terminal velocity vx: {ballistic_2_vx:.2f} m/s")

# boostback burn duration
# turning to formula notation
m0 = boostback_0_mass
delta_x_L = 1046
print(f'delta_x_L: {delta_x_L:.2f} m')
x0 = boostback_0_x
vx0 = boostback_0_vx
t0 = boostback_0_time
t2 = ballistic_2_time
T0 = T_bb

t_1_minus_0 = (2 * m0 * (delta_x_L + x0 + vx0 * (t2-t0)))/ \
        (T0 * (t2 - t0) + 2 * mdot_bb * (delta_x_L + x0 + vx0 * (t2 - t0)))
t1 = t_1_minus_0 + t0
vx_1 = vx0 + T0 / (m0 - mdot_bb * (t1 - t0)) * (t1 - t0)

print(f'Boostback time: {t_1_minus_0:.2f} s')
print(f'Terminal velocity: {vx_1:.2f} m/s')

'''
Results of iteration 1:

Boostback initial conditions at t = 101.50 s:
Mass: 1980917.90 kg
Position x: 31717.45 m
Velocity vx: 649.82 m/s
Ballistic arc terminal time: 295.95 s
Ballistic arc terminal mass: 1629148.56 kg
Ballistic arc terminal position x: 31505.61 m
Ballistic arc terminal velocity vx: -52.18 m/s
delta_x_L: 1046.00 m
Boostback time: 49.99 s
Terminal velocity: 2286.45 m/s

'''
