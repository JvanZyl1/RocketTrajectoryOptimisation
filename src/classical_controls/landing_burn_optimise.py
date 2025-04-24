import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.utils.tensorboard import SummaryWriter
from src.envs.utils.atmosphere_dynamics import gravity_model_endo, endo_atmospheric_model

def objective(callback, m0, x, sim_results=None):
    # Run simulation if results are not provided
    if sim_results is None:
        sim_results = callback.run_simulation(x)
    
    # Extract throttle profile and burn time
    u = x[:-1]
    T_burn = x[-1]
    
    # Calculate smoothness penalty for throttle profile
    smoothness_penalty = abs(np.sum(np.diff(u)**2)) / T_burn
    
    # Calculate the objective value
    # Objective is to minimize the remaining mass and smoothness penalty
    objective_value = (m0 - sim_results[2]) / 100000 + smoothness_penalty * 10
    
    return objective_value

# Callback function to print iteration information and collect data
class OptimisationCallback:
    def __init__(self,
                 simulate_function,
                 m0,
                 yf_min,
                 yf_max,
                 vf_min,
                 vf_max,
                 vy_dot_margin):
        self.iterations = []
        self.objectives = []
        self.alt_violations = []
        self.vel_violations = []
        self.dyn_violations = []
        self.prop_violations = []
        self.burn_times = []
        self.final_alts = []
        self.final_vels = []
        self.min_props = []
        self.vy_dot_violations = []  # Add list to store vy_dot violations
        self.simulation_cache = {}
        self.simulate_function = simulate_function
        self.objective_function = lambda x, sim_results=None: objective(self, m0, x, sim_results)
        self.yf_min = yf_min
        self.yf_max = yf_max
        self.vf_min = vf_min
        self.vf_max = vf_max
        self.vy_dot_margin = vy_dot_margin

        # Set up TensorBoard logging
        log_dir = 'data/reference_trajectory/landing_burn_controls/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def run_simulation(self, x):
        u = x[:-1]
        T_burn = x[-1]
        if tuple(x) not in self.simulation_cache:
            self.simulation_cache[tuple(x)] = self.simulate_function(u, T_burn)
        return self.simulation_cache[tuple(x)]

    def __call__(self, xk, state=None):
        sim_results = self.run_simulation(xk)
        y_final = sim_results[0][-1]
        v_final = sim_results[1][-1]
        min_mp = np.min(sim_results[4])
        
        y_violation = max(0, y_final - self.yf_min, self.yf_max - y_final)
        v_violation = max(0, abs(v_final) - self.vf_min, self.vf_max - abs(v_final))
        dynamic_violations = np.minimum(sim_results[3], 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        prop_violation = max(0, -min_mp)
        vy_dot_violation = max(0, abs(sim_results[5]) - self.vy_dot_margin)  # Calculate vy_dot violation
        
        # Calculate objective value
        obj_val = self.objective_function(xk, sim_results)
        
        # Store data
        self.iterations.append(len(self.iterations))
        self.objectives.append(obj_val)
        self.alt_violations.append(y_violation)
        self.vel_violations.append(v_violation)
        self.dyn_violations.append(max_dynamic_violation)
        self.prop_violations.append(prop_violation)
        self.burn_times.append(xk[-1])
        self.final_alts.append(y_final)
        self.final_vels.append(v_final)
        self.min_props.append(min_mp)
        self.vy_dot_violations.append(vy_dot_violation)  # Store vy_dot violation
        

        self.writer.add_scalar('Objective', obj_val, len(self.iterations))
        self.writer.add_scalar('Final Altitude Violation', y_violation, len(self.iterations))
        self.writer.add_scalar('Final Velocity Violation', v_violation, len(self.iterations))
        self.writer.add_scalar('Max Dynamic Pressure Violation', max_dynamic_violation, len(self.iterations))
        self.writer.add_scalar('Propellant Mass Violation', prop_violation, len(self.iterations))
        self.writer.add_scalar('Burn Time', xk[-1], len(self.iterations))
        self.writer.add_scalar('Final Altitude', y_final, len(self.iterations))
        self.writer.add_scalar('Final Velocity', v_final, len(self.iterations))
        self.writer.add_scalar('Minimum Propellant Mass', min_mp, len(self.iterations))
        self.writer.add_scalar('Vy Dot Violation', vy_dot_violation, len(self.iterations))
        return False

def simulate(u, T_end, N, y0, vy0, m0, mp0, Q_max, S_grid_fins, C_n_0, T_e, n_e, v_ex, minimum_throttle, nozzle_exit_pressure, nozzle_exit_area):
    dt = T_end / N
    y = y0
    vy = vy0
    m = m0
    mp = mp0
    t = 0
    y_vals, vy_vals, dynamic_pressure_margins, mp_vals = [], [], [], []
    for ui in u:
        ui = np.clip(ui, 0, 1) # just in case
        throttle = ui * (1 - minimum_throttle) + minimum_throttle
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        dynamic_pressure_margin = Q_max - 0.5 * density * vy**2
        dynamic_pressure_margins.append(dynamic_pressure_margin)
        force_aero = 0.5 * S_grid_fins * C_n_0 * density * vy**2
        mdot = -throttle * T_e * n_e / v_ex
        g = gravity_model_endo(y)
        T_e_losses = T_e + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area
        vy_dot = -g + throttle * T_e_losses * n_e / m + force_aero / m
        y += vy * dt
        vy += vy_dot * dt
        m += mdot * dt
        mp += mdot * dt
        t += dt
        y_vals.append(y)
        vy_vals.append(vy)
        mp_vals.append(mp)
    return np.array(y_vals), np.array(vy_vals), m, np.array(dynamic_pressure_margins), np.array(mp_vals), vy_dot

class LandingBurnOptimiser:
    def __init__(self):
        self.load_params()
        self.load_initial_conditions()
        # Discretisation
        self.N = 500             # number of time intervals

        # Bounds
        self.T_burn_min = 10.0    # minimum burn time [s]
        self.T_burn_max = 100.0    # maximum burn time [s]

        # Landing conditions
        self.yf_min = 4.75
        self.yf_max = 5.0
        self.vf_min = -0.05
        self.vf_max = 0.05
        self.vy_dot_margin = 0.01

        self.minimum_throttle = 0.4

        # Initial guess
        u0 = np.zeros(self.N)
        u0[:self.N//3] = 0.7
        u0[self.N//3:2*self.N//3] = 0.95
        u0[2*self.N//3:] = 0.95
        T_burn0 = 90.0
        self.x0 = np.append(u0, [T_burn0])

        # Bounds for throttle (0 to 1), burn time (T_burn_min to T_burn_max)
        self.bounds = [(0, 1)] * self.N + [(self.T_burn_min, self.T_burn_max)]

        self.constraints = [
            {'type': 'ineq', 'fun': self.constr_final_y},
            {'type': 'ineq', 'fun': self.constr_final_v},
            {'type': 'ineq', 'fun': self.constr_dynamic},
            {'type': 'ineq', 'fun': self.constr_propellant},
            {'type': 'ineq', 'fun': self.constr_vy_dot}
        ]
        
        # Pass the simulation function directly to the callback
        self.simulation_func_lambda = lambda u, T_end: simulate(u,
                                                                T_end,
                                                                N = self.N,
                                                                y0 = self.y0,
                                                                vy0 = self.vy0,
                                                                m0 = self.m0,
                                                                mp0 = self.mp0,
                                                                Q_max = self.Q_max,
                                                                S_grid_fins = self.S_grid_fins,
                                                                C_n_0 = self.C_n_0,
                                                                T_e = self.T_e,
                                                                n_e = self.n_e,
                                                                v_ex = self.v_ex,
                                                                minimum_throttle = self.minimum_throttle,
                                                                nozzle_exit_pressure = self.nozzle_exit_pressure,
                                                                nozzle_exit_area = self.nozzle_exit_area)
        self.callback = OptimisationCallback(self.simulation_func_lambda, self.m0, self.yf_min, self.yf_max, self.vf_min, self.vf_max, self.vy_dot_margin)
        self.objective = lambda x, sim_results=None: objective(self.callback, self.m0, x, sim_results)

    def __call__(self):
        # Solve with adjusted parameters
        result = minimize(
            self.objective, self.x0,
            method='trust-constr',
            bounds=self.bounds,
            constraints=self.constraints,
            options={
                'maxiter': 50,
                'verbose': 2,
                'gtol': 4e-1,
                'xtol': 4e-1,
                'barrier_tol': 1e-4,
                'initial_tr_radius': 1.0
            },
            callback=self.callback
        )
        self.save_all(result)
    def save_all(self, result):
        self.save_optimisation_data()
        self.plot_optimisation_history()

        # Output results
        x_opt = result.x
        u_opt = x_opt[:-1]
        T_end_opt = x_opt[-1]
        self.save_final_parameters(u_opt, T_end_opt)
        self.plot_simulation_results(u_opt, T_end_opt)

    def load_params(self):
        # Load parameters from rocket parameters
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        self.T_e = float(sizing_results['Thrust engine stage 1'])
        self.n_e = 4                                  # Select number of landing burn engines
        self.v_ex = float(sizing_results['Exhaust velocity stage 1'])
        self.Q_max = 30000.0     # max dynamic pressure [Pa]
        self.m_strc = float(sizing_results['Actual structural mass stage 1'])*1000
        self.S_grid_fins = float(sizing_results['S_grid_fins'])
        self.C_n_0 = float(sizing_results['C_n_0'])
        self.nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1'])
        self.nozzle_exit_area = float(sizing_results['Nozzle exit area'])
        # ass. alpha effective = 0 and delta = 0

    def load_initial_conditions(self):
        # Load initial conditions
        file_path_re_entry_data = 'data/reference_trajectory/re_entry_burn_controls/state_action_re_entry_burn_control.csv'
        re_entry_data = pd.read_csv(file_path_re_entry_data)
        self.y0 = re_entry_data['y[m]'].iloc[-1]    # initial altitude [m]
        self.vy0 = re_entry_data['vy[m/s]'].iloc[-1]  # initial vertical velocity [m/s]
        self.m0 = re_entry_data['mass[kg]'].iloc[-1]   # initial mass [kg]
        self.mp0 = self.m0 - self.m_strc          # propellant mass [kg]
        
    def constr_final_y(self,x):
        sim_results = self.callback.run_simulation(x)
        y_final = sim_results[0][-1]
        return np.array([y_final - self.yf_min, self.yf_max - y_final])

    def constr_final_v(self, x):
        sim_results = self.callback.run_simulation(x)
        v_final = sim_results[1][-1]
        return np.array([self.vf_min - v_final, self.vf_max + v_final])

    def constr_dynamic(self, x):
        sim_results = self.callback.run_simulation(x)
        return sim_results[3]  # must be >= 0

    def constr_propellant(self, x):
        sim_results = self.callback.run_simulation(x)
        mp_vals = sim_results[4]
        return mp_vals  # must be >= 0

    def constr_vy_dot(self, x):
        sim_results = self.callback.run_simulation(x)
        vy_dot = sim_results[5]
        # Must be close to 0
        return self.vy_dot_margin - abs(vy_dot) # must be >= 0
    
    def plot_optimisation_history(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.callback.iterations, self.callback.objectives, 'b-', label='Objective')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Objective Value')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.semilogy(self.callback.iterations, self.callback.alt_violations, 'g-', label='Altitude')
        plt.semilogy(self.callback.iterations, self.callback.vel_violations, 'r-', label='Velocity')
        plt.semilogy(self.callback.iterations, self.callback.dyn_violations, 'b-', label='Dynamic Pressure')
        plt.semilogy(self.callback.iterations, self.callback.prop_violations, 'k-', label='Propellant')
        plt.semilogy(self.callback.iterations, self.callback.vy_dot_violations, 'y-', label='Vy Dot')
        plt.xlabel('Iteration')
        plt.ylabel('Constraint Violation')
        plt.title('Constraint Violations')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/classical_controllers/landing_burn_optimisation_history.png')
        plt.close()

    def plot_simulation_results(self, u_opt, T_burn_opt):
        ys, vys, m_final, margins, mps, vy_dot  = self.simulation_func_lambda(u_opt, T_burn_opt)
        u_opt = np.clip(u_opt, 0, 1) # just in case
        throttle_opt = u_opt * (1 - self.minimum_throttle) + self.minimum_throttle
        t = np.linspace(0, T_burn_opt, self.N)

        # save data to csv
        total_masses = self.m_strc + mps
        with open('data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_simulation_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time [s]', 'Altitude [m]', 'Velocity [m/s]', 'Non-NominalThrottle', 'Propellant Mass [kg]', 'Total Mass [kg]'])
            for i in range(self.N):
                writer.writerow([t[i], ys[i], vys[i], throttle_opt[i], mps[i], total_masses[i]])
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(t, ys, 'b-')
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [m]')
        plt.title('Altitude vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(t, vys, 'r-')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(t, throttle_opt, 'g-')
        plt.xlabel('Time [s]')
        plt.ylabel('Throttle')
        plt.title('Throttle vs Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(t, mps, 'k-')
        plt.xlabel('Time [s]')
        plt.ylabel('Propellant Mass [kg]')
        plt.title('Propellant Mass vs Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/classical_controllers/landing_burn_optimisation_simulation_results.png')
        plt.close()

    def save_optimisation_data(self):
        filename = 'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_history.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Objective', 'Altitude_Violation', 'Velocity_Violation',
                            'Dynamic_Pressure_Violation', 'Propellant_Violation', 'Vy_Dot_Violation', 'Burn_Time',
                            'Final_Altitude', 'Final_Velocity', 'Min_Propellant_Mass'])
            for i in range(len(self.callback.iterations)):
                writer.writerow([
                    self.callback.iterations[i],
                    self.callback.objectives[i],
                    self.callback.alt_violations[i],
                    self.callback.vel_violations[i],
                    self.callback.dyn_violations[i],
                    self.callback.prop_violations[i],
                    self.callback.vy_dot_violations[i],
                    self.callback.burn_times[i],
                    self.callback.final_alts[i],
                    self.callback.final_vels[i],
                    self.callback.min_props[i]
                ])

    def save_final_parameters(self, u_opt, T_burn_opt):
        filename = 'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_final_parameters.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Burn_Time', T_burn_opt])
            writer.writerow(['Throttle_Profile'] + list(u_opt))
            writer.writerow(['N', self.N])


if __name__ == '__main__':
    landing_burn_optimiser = LandingBurnOptimiser()
    landing_burn_optimiser()