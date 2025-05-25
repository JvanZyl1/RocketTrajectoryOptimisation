import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from math import log as ln
import csv
import math
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from src.envs.load_initial_states import load_high_altitude_ballistic_arc_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

initial_state = load_high_altitude_ballistic_arc_initial_state()
x0, y0, vx0, vy0, theta0, theta_dot0, gamma0, alpha0, m0, mp0, time0 = initial_state

max_q = 25e3

y_refs = np.linspace(50000, 10.0, 1000)
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

print(f'Tmax/m0 * 1/g0: {Tmax/m0 * 1/g0:.6e}')

# Want to find t2 and t1 equal to 0
# Such that delta_t is minimised
def f_solve_function(t1, delta_t):
    # End of ballistic arc
    y1 = y0 + vy0 * t1 - 0.5 * g0 * t1**2
    v1 = vy0 - g0 * t1
    v2 = v1 - g0 * delta_t + Tmax/mdotmax * ln(m0 - mdotmax * delta_t)
    y2 = y1 - 0.5 * g0 * delta_t**2  + Tmax/(mdotmax**2) * (m0 - mdotmax * delta_t) * (ln(m0 - mdotmax * delta_t) - 1)
    # areq := dV/dy * dy/dt = dV/dy|y_2 * v2
    # CHECK SIGNS HERE
    a_req_RHS = -(3 * a * y2**2 + 2 * b * y2 + c) * v2
    a_req_LHS = Tmax/(m0 - mdotmax * delta_t) - g0
    sol = a_req_RHS - a_req_LHS
    return sol, (y1, y2, v1, v2)

'''
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
    return delta_t

# bounds on t1
t1 = (200, 300)
delta_t = (0, 15)
bounds = [t1, delta_t]

# initial guess
t1_init = 250
delta_t_init = 10
'''


# ---------------------------------------------------------------
# optimisation wrapper
# ---------------------------------------------------------------
from scipy.optimize import minimize, NonlinearConstraint, Bounds
objective = lambda x: x[1]                                   # minimise Δt

eq_cons   = NonlinearConstraint(lambda x: f_solve_function(x[0], x[1])[0],
                                0.0, 0.0)

ineq1     = NonlinearConstraint(lambda x: m0 - mdotmax*x[1],
                                0.0, np.inf)                 # log positivity

ineq2     = NonlinearConstraint(lambda x: 0.5*mp0 - mdotmax*x[1],
                                0.0, np.inf)                 # ≤50 % propellant

bounds    = Bounds([200.0, 0.0], [300.0, 55.0])              # t₁, Δt

x0        = np.array([250.0, 10.0])                          # initial guess

result = minimize(objective, x0,
                  method='SLSQP',
                  bounds=bounds,
                  constraints=[eq_cons, ineq1, ineq2],
                  options={'ftol': 1e-9, 'maxiter': 200, 'disp': True})

t1_opt, delta_t_opt = result.x
print(f'Optimal t1 = {t1_opt:.3f} s,   Δt = {delta_t_opt:.3f} s')