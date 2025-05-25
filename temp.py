import csv
from src.envs.load_initial_states import load_landing_burn_initial_state

state = load_landing_burn_initial_state()
x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

print(f'mass: {mass}')

sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]

number_of_engines_min = 3
minimum_engine_throttle = 0.4
nominal_throttle = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1'])
thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1'])

F_max_t = number_of_engines_gimballed * thrust_per_engine_no_losses
g0 = 9.81
a_max_g = (F_max_t/mass - g0) / g0

print(f'a_max_g: {a_max_g}')

