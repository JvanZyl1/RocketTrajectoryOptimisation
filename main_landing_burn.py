import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.utils.tensorboard import SummaryWriter

# Load parameters from rocket parameters
sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]
T_e = float(sizing_results['Thrust engine stage 1'])
n_e_start = 6                                                                 # Select number of landing burn engines
n_e_landing = 3
v_ex = float(sizing_results['Exhaust velocity stage 1'])
Q_max = 30000.0     # max dynamic pressure [Pa]
m_strc = float(sizing_results['Actual structural mass stage 1'])*1000
S_grid_fins = float(sizing_results['S_grid_fins'])
C_n_0 = float(sizing_results['C_n_0'])
# ass. alpha effective = 0 and delta = 0

# Load initial conditions
file_path_re_entry_data = 'data/reference_trajectory/re_entry_burn_controls/state_action_re_entry_burn_control.csv'
re_entry_data = pd.read_csv(file_path_re_entry_data)
y0 = re_entry_data['y[m]'].iloc[-1]    # initial altitude [m]
vy0 = re_entry_data['vy[m/s]'].iloc[-1]  # initial vertical velocity [m/s]
m0 = re_entry_data['mass[kg]'].iloc[-1]   # initial mass [kg]
mp0 = m0 - m_strc          # propellant mass [kg]

# Atmospheric parameters
g0 = 9.80665        # gravity [m/s^2]
T_base = 288.15     # ISA base temperature [K]
p_base = 101325.0   # ISA base pressure [Pa]
a_lapse = -0.0065   # temperature lapse rate [K/m]
R = 287.05          # specific gas constant [J/(kg·K)]

# Discretisation
N = 200             # number of time intervals
T_burn_min = 10.0    # minimum burn time [s]
T_burn_max = 100.0    # maximum burn time [s]
T_outer_engine_cutoff_min = 0.0
T_outer_engine_cutoff_max = 95.0

# Landing conditions
yf_min = 4.75
yf_max = 5.0
vf_min = -0.05
vf_max = 0.05
vy_dot_margin = 0.01

minimum_throttle = 0.4

# Density as function of altitude
def density(y):
    T = T_base + a_lapse * y
    p = p_base * (T / T_base) ** (g0 / (a_lapse * R))
    return p / (R * T)

def simulate(u, T_end, T_outer_engine_cutoff):
    dt = T_end / N
    y = y0
    vy = vy0
    m = m0
    mp = mp0
    t = 0
    n_e = n_e_start
    y_vals, vy_vals, dynamic_pressure_margins, mp_vals = [], [], [], []
    for ui in u:
        ui = np.clip(ui, 0, 1) # just in case
        throttle = ui * (1 - minimum_throttle) + minimum_throttle
        rho_val = density(y)
        dynamic_pressure_margin = Q_max - 0.5 * rho_val * vy**2
        dynamic_pressure_margins.append(dynamic_pressure_margin)
        force_aero = 0.5 * S_grid_fins * C_n_0 * rho_val * vy**2
        mdot = -throttle * T_e * n_e / v_ex
        vy_dot = -g0 + throttle * T_e * n_e / m + force_aero / m
        y += vy * dt
        vy += vy_dot * dt
        m += mdot * dt
        mp += mdot * dt
        t += dt
        if t > T_outer_engine_cutoff:
            n_e = n_e_landing
        y_vals.append(y)
        vy_vals.append(vy)
        mp_vals.append(mp)
    return np.array(y_vals), np.array(vy_vals), m, np.array(dynamic_pressure_margins), np.array(mp_vals), vy_dot

# Objective: minimise propellant used
def objective(x, sim_results=None):
    if sim_results is None:
        sim_results = callback.run_simulation(x)
    u = x[:-2] # throttle
    t_burn = x[-2]
    # Regularisation smoother
    smoothness_penalty = abs(np.sum(np.diff(u)**2))/t_burn
    return (m0 - sim_results[2])/1000 + smoothness_penalty*20

# Constraints
def constr_final_y(x):
    sim_results = callback.run_simulation(x)
    y_final = sim_results[0][-1]
    return np.array([y_final - yf_min, yf_max - y_final])

def constr_final_v(x):
    sim_results = callback.run_simulation(x)
    v_final = sim_results[1][-1]
    return np.array([vf_min - v_final, vf_max + v_final])

def constr_dynamic(x):
    sim_results = callback.run_simulation(x)
    return sim_results[3]  # must be >= 0

def constr_propellant(x):
    sim_results = callback.run_simulation(x)
    mp_vals = sim_results[4]
    return mp_vals  # must be >= 0

def constr_vy_dot(x):
    u = x[:-2]
    T_burn = x[-2]
    T_outer_engine_cutoff = x[-1]
    sim_results = callback.run_simulation(x)
    vy_dot = sim_results[5]
    # Must be close to 0
    return (vy_dot_margin - abs(vy_dot))*2 # must be >= 0

# Callback function to print iteration information and collect data
class OptimizationCallback:
    def __init__(self):
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

        # Set up TensorBoard logging
        log_dir = 'data/reference_trajectory/landing_burn_controls/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def run_simulation(self, x):
        u = x[:-2]
        T_burn = x[-2]
        T_outer_engine_cutoff = x[-1]
        if tuple(x) not in self.simulation_cache:
            self.simulation_cache[tuple(x)] = simulate(u, T_burn, T_outer_engine_cutoff)
        return self.simulation_cache[tuple(x)]

    def __call__(self, xk, state=None):
        sim_results = self.run_simulation(xk)
        y_final = sim_results[0][-1]
        v_final = sim_results[1][-1]
        min_mp = np.min(sim_results[4])
        
        y_violation = max(0, y_final - yf_min, yf_max - y_final)
        v_violation = max(0, abs(v_final) - vf_min, vf_max - abs(v_final))
        dynamic_violations = np.minimum(sim_results[3], 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        prop_violation = max(0, -min_mp)
        vy_dot_violation = max(0, abs(sim_results[5]) - vy_dot_margin)  # Calculate vy_dot violation
        
        # Calculate objective value
        obj_val = objective(xk, sim_results)
        
        # Store data
        self.iterations.append(len(self.iterations))
        self.objectives.append(obj_val)
        self.alt_violations.append(y_violation)
        self.vel_violations.append(v_violation)
        self.dyn_violations.append(max_dynamic_violation)
        self.prop_violations.append(prop_violation)
        self.burn_times.append(xk[-2])
        self.final_alts.append(y_final)
        self.final_vels.append(v_final)
        self.min_props.append(min_mp)
        self.vy_dot_violations.append(vy_dot_violation)  # Store vy_dot violation
        
        # Log data to TensorBoard every 10 iterations
        self.writer.add_scalar('Objective', obj_val, len(self.iterations))
        self.writer.add_scalar('Final Altitude Violation', y_violation, len(self.iterations))
        self.writer.add_scalar('Final Velocity Violation', v_violation, len(self.iterations))
        self.writer.add_scalar('Max Dynamic Pressure Violation', max_dynamic_violation, len(self.iterations))
        self.writer.add_scalar('Propellant Mass Violation', prop_violation, len(self.iterations))
        self.writer.add_scalar('Burn Time', xk[-2], len(self.iterations))
        self.writer.add_scalar('Outer Engine Cutoff Time', xk[-1], len(self.iterations))
        self.writer.add_scalar('Final Altitude', y_final, len(self.iterations))
        self.writer.add_scalar('Final Velocity', v_final, len(self.iterations))
        self.writer.add_scalar('Minimum Propellant Mass', min_mp, len(self.iterations))
        self.writer.add_scalar('Vy Dot Violation', vy_dot_violation, len(self.iterations))  # Log vy_dot violation
        '''
        print(f"\nIteration Information:")
        print(f"Burn time: {xk[-2]:.2f} s")
        print(f"Outer engine cutoff time: {xk[-1]:.2f} s")
        print(f"Final altitude: {y_final:.3f} m")
        print(f"Final velocity: {v_final:.3f} m/s")
        print(f"Minimum propellant mass: {min_mp:.2f} kg")
        print(f"Objective value: {obj_val:.2f}")
        print(f"Final altitude constraint violation: {y_violation:.3e} m")
        print(f"Final velocity constraint violation: {v_violation:.3e} m/s")
        print(f"Maximum dynamic pressure constraint violation: {max_dynamic_violation:.2e} Pa")
        print(f"Propellant mass constraint violation: {prop_violation:.2e} kg")
        print(f"Vy dot constraint violation: {vy_dot_violation:.3e} m/s")
        '''
        return False

def plot_optimization_history(callback):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(callback.iterations, callback.objectives, 'b-', label='Objective')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Value')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(callback.iterations, callback.alt_violations, 'g-', label='Altitude')
    plt.semilogy(callback.iterations, callback.vel_violations, 'r-', label='Velocity')
    plt.semilogy(callback.iterations, callback.dyn_violations, 'b-', label='Dynamic Pressure')
    plt.semilogy(callback.iterations, callback.prop_violations, 'k-', label='Propellant')
    plt.semilogy(callback.iterations, callback.vy_dot_violations, 'y-', label='Vy Dot')
    plt.xlabel('Iteration')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/classical_controllers/landing_burn_optimisation_history.png')
    plt.close()

def plot_simulation_results(u_opt, T_burn_opt, T_outer_engine_cutoff_opt):
    ys, vys, m_final, margins, mps, vy_dot = simulate(u_opt, T_burn_opt, T_outer_engine_cutoff_opt)
    u_opt = np.clip(u_opt, 0, 1) # just in case
    throttle_opt = u_opt * (1 - minimum_throttle) + minimum_throttle
    t = np.linspace(0, T_burn_opt, N)

    # save data to csv
    total_masses = m_strc + mps
    with open('data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_simulation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time [s]', 'Altitude [m]', 'Velocity [m/s]', 'Non-NominalThrottle', 'Propellant Mass [kg]', 'Total Mass [kg]'])
        for i in range(N):
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

def save_optimization_data(callback, filename='optimization_history.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Objective', 'Altitude_Violation', 'Velocity_Violation',
                        'Dynamic_Pressure_Violation', 'Propellant_Violation', 'Vy_Dot_Violation', 'Burn_Time',
                        'Final_Altitude', 'Final_Velocity', 'Min_Propellant_Mass'])
        for i in range(len(callback.iterations)):
            writer.writerow([
                callback.iterations[i],
                callback.objectives[i],
                callback.alt_violations[i],
                callback.vel_violations[i],
                callback.dyn_violations[i],
                callback.prop_violations[i],
                callback.vy_dot_violations[i],
                callback.burn_times[i],
                callback.final_alts[i],
                callback.final_vels[i],
                callback.min_props[i]
            ])

def save_final_parameters(u_opt, T_burn_opt, T_outer_engine_cutoff_opt, filename='final_parameters.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Burn_Time', T_burn_opt])
        writer.writerow(['Outer_Engine_Cutoff_Time', T_outer_engine_cutoff_opt])
        writer.writerow(['Throttle_Profile'] + list(u_opt))

# Initial guess
u0 = np.zeros(N)
u0[:N//3] = 0.4
u0[N//3:2*N//3] = 0.6
u0[2*N//3:] = 0.95
T_burn0 = 88.0
T_outer_engine_cutoff0 = 55.0
x0 = np.append(u0, [T_burn0, T_outer_engine_cutoff0])

# Bounds for throttle (0 to 1), burn time (T_burn_min to T_burn_max), and outer engine cutoff time (T_outer_engine_cutoff_min to T_outer_engine_cutoff_max)
bounds = [(0, 1)] * N + [(T_burn_min, T_burn_max), (T_outer_engine_cutoff_min, T_outer_engine_cutoff_max)]

constraints = [
    {'type': 'ineq', 'fun': constr_final_y},
    {'type': 'ineq', 'fun': constr_final_v},
    {'type': 'ineq', 'fun': constr_dynamic},
    {'type': 'ineq', 'fun': constr_propellant},
    {'type': 'ineq', 'fun': constr_vy_dot}
]

# Create callback instance
callback = OptimizationCallback()

# Solve with adjusted parameters
result = minimize(
    objective, x0,
    method='trust-constr',
    bounds=bounds,
    constraints=constraints,
    options={
        'maxiter': 50,
        'verbose': 2,
        'gtol': 4e-4,
        'xtol': 4e-4,
        'barrier_tol': 1e-4,
        'initial_tr_radius': 1.0
    },
    callback=callback
)

# Save and plot results
save_optimization_data(callback,
                       f'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_history.csv')
plot_optimization_history(callback)

# Output results
x_opt = result.x
u_opt = x_opt[:-2]
T_end_opt = x_opt[-2]
T_outer_engine_cutoff_opt = x_opt[-1]
save_final_parameters(u_opt, T_end_opt, T_outer_engine_cutoff_opt,
                      f'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_final_parameters.csv')
plot_simulation_results(u_opt, T_end_opt, T_outer_engine_cutoff_opt)