import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from math import log as ln
import csv
import math
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import pyswarms as ps
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

# Read 
data = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
y_ball = data['y[m]']
vy_ball = data['vy[m/s]']
m_ball = data['mass[kg]']
mp_ball = data['mass_propellant[kg]']
# Interpolate to find vy, m and mp at y = 55km
y0 = 55000
vy0 = np.interp(y0, y_ball, vy_ball)
m0 = np.interp(y0, y_ball, m_ball)
mp0 = np.interp(y0, y_ball, mp_ball)


max_q = 25e3

y_refs = np.linspace(55000, 10.0, 1000)
max_v_s = np.zeros(len(y_refs))
no_thrust_velocities = np.zeros(len(y_refs))
for i, y_ref in enumerate(y_refs):
    air_densities, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_ref)
    if math.isnan(air_densities):
        print(f"Air density is NaN at altitude {y_ref} m")
    if air_densities == 0:
        print(f"Air density is 0 at altitude {y_ref} m")
    max_v_s[i] = np.sqrt(2 * max_q /air_densities)

v_max_fcn = np.poly1d(np.polyfit(y_refs, max_v_s, 3))

# Check fit
# average max v error
avg_error = np.mean(np.abs(v_max_fcn(y_refs) - max_v_s))
avg_percent_error = np.mean(np.abs(v_max_fcn(y_refs) - max_v_s) / max_v_s) * 100
print(f"Average max velocity error: {avg_error:.2f} m/s")
print(f"Average max velocity percent error: {avg_percent_error:.2f}%")

'''
alt from 50000 to 10 m

Polyfit of 3 degrees:
Average max velocity error: 19.94 m/s
Average max velocity percent error: 3.31%

Polyfit of 4 degrees
Average max velocity error: 6.10 m/s
Average max velocity percent error: 0.59%
'''

# Then plot
plt.figure(figsize=(10, 6))
plt.plot(y_refs, max_v_s, 'b-', label='Max velocity from dynamic pressure')
plt.plot(y_refs, v_max_fcn(y_refs), 'r--', label=f'Polynomial fit (order 3)')
plt.xlabel('Altitude (m)')
plt.ylabel('Maximum velocity (m/s)')
plt.title('Maximum Velocity vs Altitude for Dynamic Pressure Constraint')
plt.legend()
plt.grid(True)
plt.savefig('max_velocity_fit.png')
plt.close()

# Coefficients
a = v_max_fcn.coeffs[0] # y^3
b = v_max_fcn.coeffs[1] # y^2
c = v_max_fcn.coeffs[2] # y
d = v_max_fcn.coeffs[3] # constant

g0 = 9.81

sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]
m_burn_out = float(sizing_results['Structural mass stage 1 (ascent)'])*1000
T_e = float(sizing_results['Thrust engine stage 1'])
v_ex = float(sizing_results['Exhaust velocity stage 1'])
n_e = float(sizing_results['Number of engines gimballed stage 1'])
mdotmax = T_e/v_ex * n_e
Tmax = T_e * n_e

print(f'Tmax: {Tmax:.2f} N')
print(f'mdotmax: {mdotmax:.2f} kg/s')
print(f'm0: {m0:.2f} kg')

print(f'Tmax/m0 * 1/g0: {Tmax/m0 * 1/g0:.6e}')

# Want to find t2 and t1 equal to 0
# Such that delta_t is minimised
def f_solve_function(t1, delta_t):
    # End of ballistic arc
    v1 = vy0 - g0 * t1
    y1 = y0 + vy0 * t1 - 0.5 * g0 * t1**2
    # Burn
    N = 200
    dt = delta_t/N
    times = np.linspace(0, delta_t, N)
    m = m0
    v = v1
    y = y1
    for t in times:
        a = Tmax/m - g0
        v += a * dt
        y += v * dt
        m -= mdotmax * dt
    y2 = y
    v2 = v
    # areq := dV/dy * dy/dt = dV/dy|y_2 * v2
    # CHECK SIGNS HERE
    a_req_RHS = (3 * a * y2**2 + 2 * b * y2 + c) * v2
    a_req_LHS = Tmax/(m0 - mdotmax * delta_t) - g0
    sol = a_req_RHS - a_req_LHS
    return sol, (y1, y2, v1, v2)

delta_t2_t1_max = mp0/mdotmax * 0.5 # max 50% of prop consumed.

# Logarithmic constraint : delta_t < m0/mdotmax
def log_constraint(x):
    t1, delta_t = x
    return m0 - mdotmax * delta_t # > 0
# Propellant consumption constraint : delta_t < 0.5 * mp / mdotmax
def prop_constraint(x):
    t1, delta_t = x
    return mp0*0.5 - mdotmax * delta_t # > 0
# fsolve function
def solve_eq_constraints(x):
    t1, delta_t = x
    return f_solve_function(t1, delta_t) # = 0

constraints = {
    'type': 'eq', 'fun': solve_eq_constraints,
    'type': 'ineq', 'fun': log_constraint,
    'type': 'ineq', 'fun': prop_constraint,
}

# Objective is to minimise delta_t
def objective(x):
    t1, delta_t = x
    penalty = 0.0
    log_const = log_constraint(x)
    if log_const < 0:
        penalty += 500 * abs(log_const)
    if penalty == 0:
        residual, _aux_values = f_solve_function(t1, delta_t)
        y1, y2, v1, v2 = _aux_values
        if y1 < 0:
            penalty += abs(y1)
        if y2 < 0:
            penalty += abs(y2)
        if abs(residual) > 1e-3:
            penalty += abs(residual)*100
    prop_const = prop_constraint(x)
    if prop_const < 0:
        penalty += 100 * abs(prop_const)
    reward = delta_t - penalty
    return -reward

# bounds on t1
t1 = (0, 20)
delta_t = (0, 50)
bounds = [t1, delta_t]

# initial guess
t1_init = 250
delta_t_init = 10

# Use pyswarms to solve

# PySwarms wrapper for the objective function (needs to handle multiple particles)
def pyswarms_objective(x):
    n_particles = x.shape[0]
    j = np.zeros(n_particles)
    
    for i in range(n_particles):
        j[i] = objective([x[i, 0], x[i, 1]])
    
    return j

# Set up bounds for pyswarms
lb = np.array([t1[0], delta_t[0]])
ub = np.array([t1[1], delta_t[1]])
bounds = (lb, ub)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(pyswarms_objective, iters=1000, verbose=True)

# Extract best solution
best_t1, best_delta_t = pos

# Calculate trajectory parameters at key points
sol, (y1, y2, v1, v2) = f_solve_function(best_t1, best_delta_t)

print("\nPySwarms Optimization Results:")
print(f"Best t1: {best_t1:.4f} seconds")
print(f"Best delta_t: {best_delta_t:.4f} seconds")
print(f"Solution residual: {sol:.6e}")
print(f"Total time: {best_t1 + best_delta_t:.4f} seconds")
print(f"Propellant consumed: {mdotmax * best_delta_t:.2f} kg ({mdotmax * best_delta_t / mp0 * 100:.2f}% of available)")

print("\nTrajectory Parameters:")
print(f"Initial altitude: {y0:.2f} m, velocity: {vy0:.2f} m/s")
print(f"At t1 (end of free-fall): altitude: {y1:.2f} m, velocity: {v1:.2f} m/s")
print(f"At t2 (end of burn): altitude: {y2:.2f} m, velocity: {v2:.2f} m/s")
print(f"Max allowable velocity at final altitude: {v_max_fcn(y2):.2f} m/s")


