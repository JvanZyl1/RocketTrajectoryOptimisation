import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.base_environment import load_landing_burn_initial_state

g_0 = 9.80665 # [m/s^2]
dt = 0.01 # [s]
max_q = 30000 # [Pa]

state_initial = load_landing_burn_initial_state()
y_0 = state_initial[1]
v_x_0 = state_initial[2]
v_y_0 = state_initial[3]
v_0 = np.sqrt(v_x_0**2 + v_y_0**2)

y_refs = np.linspace(y_0, 0, 100)
air_densities = np.zeros(len(y_refs))
max_v_s = np.zeros(len(y_refs))
no_thrust_velocities = np.zeros(len(y_refs))
vy_no_thrust = np.zeros(len(y_refs))
for i, y_ref in enumerate(y_refs):
    air_densities[i], atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_ref)
    max_v_s[i] = np.sqrt(2* max_q /air_densities[i])
    vy_no_thrust[i] = math.sqrt(v_y_0**2 + 2*g_0*(y_0 - y_ref))

v_max_fcn = np.poly1d(np.polyfit(y_refs, max_v_s, 4))

# Define a second-order polynomial for velocity as a function of altitude
# v(y) = a*y^2 + b*y + c
# Initial conditions: v(0) = 0, v(y_0) = abs(v_y_0)
# Using v(0) = 0 => c = 0, so v(y) = a*y^2 + b*y
def compute_optimal_trajectory():
    y_samples = np.linspace(0, y_0, 200)
    def v_opt_profile(y, params):
        a, b = params
        return a * y**2 + b * y
        
    def objective(params):
        a, b = params
        return -((a/3) * y_0**3 + (b/2) * y_0**2) # maximise area under velocity curve
    
    def constraint_initial_velocity(params):
        return v_opt_profile(y_0, params) - abs(v_y_0)
    
    def constraint_velocity_limit(params):
        return v_max_fcn(y_samples) - v_opt_profile(y_samples, params) # vy < vy_lim
    
    result = minimize(
        objective,
        [0, abs(v_y_0) / y_0], # linear profile from (0,0) to (y_0, abs(v_y_0))
        constraints=[{'type': 'eq', 'fun': constraint_initial_velocity},
                     {'type': 'ineq', 'fun': constraint_velocity_limit}],
        method='SLSQP',
        options={'disp': False}  # Turn off verbose output
    )
    a_opt, b_opt = result.x
    print(f"Optimal velocity profile: v(y) = {a_opt:.6e}*y^2 + {b_opt:.6e}*y")
    v_opt = lambda y: a_opt * y**2 + b_opt * y
    return v_opt

v_opt_fn = compute_optimal_trajectory()
# ------ PLOTTING ------
y_vals = np.linspace(min(y_refs), max(y_refs), 500)
v_opt_plot = v_opt_fn(y_vals)
a_opt_plot = np.gradient(v_opt_plot, y_vals) * v_opt_plot # Chain rule dv/dt = dv/dy * dy/dt = dv/dy * v
a_max_v_s = np.gradient(v_max_fcn(y_vals), y_vals) * v_max_fcn(y_vals)

plt.figure(figsize=(20, 10))
plt.suptitle('Optimal Feasible Landing Trajectory', fontsize=22)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
ax1 = plt.subplot(gs[0])
ax1.plot(y_refs/1000, max_v_s/1000, color='blue', linewidth=4, label='Max Velocity Limit')
ax1.plot(y_vals/1000, v_max_fcn(y_vals)/1000, '--', color='grey', label='Polyfit (Limit)')
ax1.plot(y_vals/1000, v_opt_plot/1000, 'r-', linewidth=3, label='Optimal Trajectory')
ax1.scatter(y_0/1000, abs(v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
ax1.scatter(y_0/1000, abs(v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
ax1.plot(y_refs/1000, vy_no_thrust/1000, color='black', linewidth=2, linestyle='--', label='No Thrust Velocity')
ax1.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
ax1.set_ylabel(r'v [$km/s$]', fontsize=20)
ax1.set_ylim(0, 2)
ax1.set_title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory', fontsize=20)
ax1.grid(True)
ax1.legend(fontsize=20)
ax1.tick_params(labelsize=16)

ax2 = plt.subplot(gs[1])
ax2.plot(y_vals/1000, a_opt_plot/g_0, 'r-', linewidth=3, label='Generated Acceleration')
ax2.plot(y_vals/1000, a_max_v_s/g0, 'b--', linewidth=3, label='Optimal Acceleration')
ax2.set_xlabel(r'y [$km$]', fontsize=20)
ax2.set_ylabel(r'a [$g_0$]', fontsize=20)
ax2.set_title('True Acceleration (dv/dt) vs. Altitude', fontsize=20)
ax2.grid(True)
ax2.legend(fontsize=20)
ax2.tick_params(labelsize=16)
ax2.set_ylim(0, 5)

plt.savefig('results/landing_burn_optimal/initial_velocity_profile_guess.png')
plt.show()


#  -------- REFERENCE TRAJECTORY --------
# a(t) = T/m(t) * tau(t) - g_0 + 0.5 * rho(y(t)) * v(t)^2 * C_n_0 * S
# m(t) = m_0 - mdot * int_0^t tau(t) dt
# v(t) = v_0 + int_0^t a(t) dt
# y(t) = y_0 + int_0^t v(t) dt
# Constraints : m > ms, v(t) < v_max(y(t))
# Initial conditions : m(0) = m_0, v(0) = v_0, y(0) = y_0
# Final conditions : v(t_f) = 0, y(t_f) = 0
# tau(t) = (0,1)

class reference_landing_trajectory:
    def __init__(self):
        # Initial conditions
        self.m = 
        self.y
        self.v

        # Constants
        Te = 
        ne = 
        self.T = Te * ne
        v_ex
        self.mdot_max = self.T/v_ex
        self.g_0 = 9.80665
        self.C_n_0 = 3
        self.S_grid_fin = 2
        self.n_grid_fin = 4
        self.dt = 0.1
        
        # Logging
        self.a_vals = []
        self.m_vals = []
        self.y_vals = []
        self.v_vals = []

    def find_tau(self, a_des):
        tau = self.m/self.T(a_des + self.g_0 - 0.5 * endo_atmospheric_model(self.y)[0] * self.v**2 * self.C_n_0 * self.S_grid_fin * self.n_grid_fin)
        return tau  
        
    def simulation_step(self, tau):
        a = self.T/self.m * tau - self.g_0 + 0.5 * endo_atmospheric_model(self.y)[0] * self.v**2 * self.C_n_0 * self.S_grid_fin * self.n_grid_fin
        self.m = self.m - self.mdot_max * tau * self.dt
        self.v = self.v + a * self.dt
        self.y = self.y + self.v * self.dt

        self.a_vals.append(a)
        self.m_vals.append(self.m)
        self.y_vals.append(self.y)
        self.v_vals.append(self.v)

'''
Guide to finish
1) Fix reference landing trajectory
    - From the initial guess of optimal acceleration calculate the tau required.
    - Step simulation to store reference
    - Save reference to a csv
2) casadi landing burn
    - Load reference to initial guess
    - Solve tau
GENERAL) fix directly where these are all placed.
3) fix cop so stable, and find parameter variation limits at same time.
4) ACS proper sizing
5) Fix CL and CD
6) Flip over terminal vx
7) Landing burn controllers.
8) Iterate
9) Run reinforcement learning with disturbances to match reference, not controller output.
10) Bring in static parameter variations.
11) Bring in fuel sloshing
'''
        