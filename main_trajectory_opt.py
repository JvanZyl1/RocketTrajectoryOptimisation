import csv
import scipy.interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.base_environment import load_landing_burn_initial_state

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
        # Constants
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        Te = float(sizing_results['Thrust engine stage 1'])
        ne = 12
        self.T = Te * ne
        v_ex =  float(sizing_results['Exhaust velocity stage 1'])
        self.mdot_max = self.T/v_ex
        self.g_0 = 9.80665
        self.C_n_0 = 3
        self.S_grid_fin = 2
        self.n_grid_fin = 4
        self.dt = 0.1
        self.max_q = 30000 # [Pa]
        self.m_s = float(sizing_results['Structural mass stage 1 (descent)'])*1000
        
        # Logging
        self.a_vals = []
        self.m_vals = []
        self.y_vals = []
        self.v_vals = []
        self.tau_vals = []
        self.time_vals = []
        self.time = 0.0

        self.load_initial_conditions()
        self.find_dynamic_pressure_limited_velocities()
        self.compute_optimal_trajectory()
        self.post_process_results()
        
    def load_initial_conditions(self):
        state_initial = load_landing_burn_initial_state()
        self.y_0 = state_initial[1]
        v_x_0 = state_initial[2]
        self.v_y_0 = state_initial[3]
        self.v_0 = np.sqrt(v_x_0**2 + self.v_y_0**2)
        self.y_refs = np.linspace(self.y_0, 0, 100)
        # Initial conditions
        self.m = state_initial[8]
        self.y = self.y_0
        self.v = self.v_0

    def find_dynamic_pressure_limited_velocities(self):
        air_densities = np.zeros(len(self.y_refs))
        max_v_s = np.zeros(len(self.y_refs))
        no_thrust_velocities = np.zeros(len(self.y_refs))
        for i, y_ref in enumerate(self.y_refs):
            air_densities[i], atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_ref)
            max_v_s[i] = np.sqrt(2 * self.max_q /air_densities[i])

        self.v_max_fcn = np.poly1d(np.polyfit(self.y_refs, max_v_s, 4))

    # Define a second-order polynomial for velocity as a function of altitude
    # v(y) = a*y^2 + b*y + c
    # Initial conditions: v(0) = 0, v(y_0) = abs(v_y_0)
    # Using v(0) = 0 => c = 0, so v(y) = a*y^2 + b*y
    def compute_optimal_trajectory(self):
        y_samples = np.linspace(0, self.y_0, 200)
        def v_opt_profile(y, params):
            a, b = params
            return a * y**2 + b * y
            
        def objective(params):
            a, b = params
            return -((a/3) * self.y_0**3 + (b/2) * self.y_0**2) # maximise area under velocity curve
        
        def constraint_initial_velocity(params):
            return v_opt_profile(self.y_0, params) - abs(self.v_y_0)
        
        def constraint_velocity_limit(params):
            return self.v_max_fcn(y_samples) - v_opt_profile(y_samples, params) # vy < vy_lim
        
        result = minimize(
            objective,
            [0, abs(self.v_y_0) / self.y_0], # linear profile from (0,0) to (y_0, abs(v_y_0))
            constraints=[{'type': 'eq', 'fun': constraint_initial_velocity},
                        {'type': 'ineq', 'fun': constraint_velocity_limit}],
            method='trust-constr',
            options={'disp': True}
        )
        a_opt, b_opt = result.x
        print(f"Optimal velocity profile: v(y) = {a_opt:.6e}*y^2 + {b_opt:.6e}*y")
        v_opt = lambda y: a_opt * y**2 + b_opt * y
        return v_opt

    def post_process_results(self):
        v_opt_fn = self.compute_optimal_trajectory()
        # ------ PLOTTING ------
        self.y_vals_plot = np.linspace(min(self.y_refs), max(self.y_refs), 1500)
        v_opt_plot = v_opt_fn(self.y_vals_plot)
        self.a_opt_plot = np.gradient(v_opt_plot, self.y_vals_plot) * v_opt_plot # Chain rule dv/dt = dv/dy * dy/dt = dv/dy * v
        self.a_max_v_s = np.gradient(self.v_max_fcn(self.y_vals_plot), self.y_vals_plot) * self.v_max_fcn(self.y_vals_plot)

        plt.figure(figsize=(20, 10))
        plt.suptitle('Optimal Feasible Landing Trajectory', fontsize=22)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, color='blue', linewidth=4, label='Max Velocity Limit')
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, linestyle = '--', color='grey', label='Polyfit (Limit)')
        ax1.plot(self.y_vals_plot/1000, v_opt_plot/1000, color = 'red', linewidth=3, label='Initial Guess Trajectory')
        ax1.scatter(self.y_0/1000, abs(self.v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
        ax1.scatter(self.y_0/1000, abs(self.v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
        ax1.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
        ax1.set_ylabel(r'v [$km/s$]', fontsize=20)
        ax1.set_ylim(0, 2)
        ax1.set_title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory', fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=20)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.y_vals_plot/1000, self.a_opt_plot/self.g_0, 'r-', linewidth=3, label='Initial Guess Acceleration')
        ax2.plot(self.y_vals_plot/1000, self.a_max_v_s/self.g_0, 'b--', linewidth=3, label='For maximum dynamic pressure')
        ax2.set_xlabel(r'y [$km$]', fontsize=20)
        ax2.set_ylabel(r'a [$g_0$]', fontsize=20)
        ax2.set_title('True Acceleration (dv/dt) vs. Altitude', fontsize=20)
        ax2.grid(True)
        ax2.legend(fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_ylim(0, 5)

        plt.savefig('results/landing_burn_optimal/initial_velocity_profile_guess.png')
        plt.close()

        # Save y_vals_plot and v_opt_plot
        # using pandas
        df_reference = pd.DataFrame({
            'altitude': self.y_vals_plot,
            'velocity': v_opt_plot,
            'acceleration': self.a_opt_plot
        })
        df_reference.to_csv('data/reference_trajectory/landing_burn_optimal/initial_guessreference_profile.csv', index=False)

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

if __name__ == '__main__':
    ref = reference_landing_trajectory()
    