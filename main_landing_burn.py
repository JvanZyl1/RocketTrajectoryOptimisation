import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch.utils.tensorboard import SummaryWriter
import os

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
N = 500             # number of time intervals
T_burn_min = 10.0    # minimum burn time [s]
T_burn_max = 100.0    # maximum burn time [s]
T_outer_engine_cutoff_min = 0.0
T_outer_engine_cutoff_max = 95.0

# Landing conditions
yf_min = 4.75
yf_max = 5.0
vf_min = -0.05
vf_max = 0.05

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
    return np.array(y_vals), np.array(vy_vals), m, np.array(dynamic_pressure_margins), np.array(mp_vals)

# Objective: minimise propellant used
def objective(x):
    u = x[:-2] # throttle
    T_burn = x[-2] # burn time
    T_outer_engine_cutoff = x[-1] # outer engine cutoff time
    sim = simulate(u, T_burn, T_outer_engine_cutoff)

    # Regularisation smoother
    smoothness_penalty = abs(np.sum(np.diff(u)**2))

    return (m0 - sim[2])/1000 + smoothness_penalty

# Constraints
def constr_final_y(x):
    u = x[:-2]
    T_burn = x[-2]
    T_outer_engine_cutoff = x[-1]
    y_final = simulate(u, T_burn, T_outer_engine_cutoff)[0][-1]
    return np.array([y_final - yf_min, yf_max - y_final])

def constr_final_v(x):
    u = x[:-2]
    T_burn = x[-2]
    T_outer_engine_cutoff = x[-1]
    v_final = simulate(u, T_burn, T_outer_engine_cutoff)[1][-1]
    return np.array([vf_min - v_final, vf_max + v_final])

def constr_dynamic(x):
    u = x[:-2]
    T_burn = x[-2]
    T_outer_engine_cutoff = x[-1]
    return simulate(u, T_burn, T_outer_engine_cutoff)[3]  # must be >= 0

def constr_propellant(x):
    u = x[:-2]
    T_burn = x[-2]
    T_outer_engine_cutoff = x[-1]
    mp_vals = simulate(u, T_burn, T_outer_engine_cutoff)[4]
    return mp_vals  # must be >= 0

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
        log_dir = 'data/reference_trajectory/landing_burn_controls/logs'
        # Set up TensorBoard logging
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def __call__(self, xk, state=None):
        u = xk[:-2]
        T_burn = xk[-2]
        T_outer_engine_cutoff = xk[-1]
        # Calculate constraint violations
        y_final = simulate(u, T_burn, T_outer_engine_cutoff)[0][-1]
        v_final = simulate(u, T_burn, T_outer_engine_cutoff)[1][-1]
        mps = simulate(u, T_burn, T_outer_engine_cutoff)[4]
        min_mp = np.min(mps)
        
        y_violation = max(0, y_final - yf_min, yf_max - y_final)
        v_violation = max(0, abs(v_final) - vf_min, vf_max - abs(v_final))
        dynamic_violations = np.minimum(constr_dynamic(xk), 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        prop_violation = max(0, -min_mp)
        
        # Calculate objective value
        obj_val = objective(xk)
        smoothness_penalty = abs(np.sum(np.diff(u)**2))
        
        # Store data
        self.iterations.append(len(self.iterations))
        self.objectives.append(obj_val)
        self.alt_violations.append(y_violation)
        self.vel_violations.append(v_violation)
        self.dyn_violations.append(max_dynamic_violation)
        self.prop_violations.append(prop_violation)
        self.burn_times.append(T_burn)
        self.final_alts.append(y_final)
        self.final_vels.append(v_final)
        self.min_props.append(min_mp)
        
        # Log data to TensorBoard
        self.writer.add_scalar('Objective', obj_val, len(self.iterations))
        self.writer.add_scalar('Final Altitude Violation', y_violation, len(self.iterations))
        self.writer.add_scalar('Final Velocity Violation', v_violation, len(self.iterations))
        self.writer.add_scalar('Max Dynamic Pressure Violation', max_dynamic_violation, len(self.iterations))
        self.writer.add_scalar('Propellant Mass Violation', prop_violation, len(self.iterations))
        self.writer.add_scalar('Burn Time', T_burn, len(self.iterations))
        self.writer.add_scalar('Final Altitude', y_final, len(self.iterations))
        self.writer.add_scalar('Final Velocity', v_final, len(self.iterations))
        self.writer.add_scalar('Minimum Propellant Mass', min_mp, len(self.iterations))
        self.writer.add_scalar('Smoothness Penalty', smoothness_penalty, len(self.iterations))
        print(f"\nIteration Information:")
        print(f"Burn time: {T_burn:.2f} s")
        print(f"Outer engine cutoff time: {T_outer_engine_cutoff:.2f} s")
        print(f"Final altitude: {y_final:.3f} m")
        print(f"Final velocity: {v_final:.3f} m/s")
        print(f"Minimum propellant mass: {min_mp:.2f} kg")
        print(f"Objective value: {obj_val:.2f}")
        print(f"Final altitude constraint violation: {y_violation:.3e} m")
        print(f"Final velocity constraint violation: {v_violation:.3e} m/s")
        print(f"Maximum dynamic pressure constraint violation: {max_dynamic_violation:.2e} Pa")
        print(f"Propellant mass constraint violation: {prop_violation:.2e} kg")
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
    plt.xlabel('Iteration')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/classical_controllers/landing_burn_optimisation_history.png')
    plt.close()

def plot_simulation_results(u_opt, T_burn_opt, T_outer_engine_cutoff_opt):
    ys, vys, m_final, margins, mps = simulate(u_opt, T_burn_opt, T_outer_engine_cutoff_opt)
    u_opt = np.clip(u_opt, 0, 1) # just in case
    throttle_opt = u_opt * (1 - minimum_throttle) + minimum_throttle
    t = np.linspace(0, T_burn_opt, N)
    
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
                        'Dynamic_Pressure_Violation', 'Propellant_Violation', 'Burn_Time',
                        'Final_Altitude', 'Final_Velocity', 'Min_Propellant_Mass'])
        for i in range(len(callback.iterations)):
            writer.writerow([
                callback.iterations[i],
                callback.objectives[i],
                callback.alt_violations[i],
                callback.vel_violations[i],
                callback.dyn_violations[i],
                callback.prop_violations[i],
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
u0[:N//3] = 0.7
u0[N//3:2*N//3] = 0.7
u0[2*N//3:] = 0.8
T_burn0 = 75.0
T_outer_engine_cutoff0 = 25.0
x0 = np.append(u0, [T_burn0, T_outer_engine_cutoff0])

# Bounds for throttle (0 to 1), burn time (T_burn_min to T_burn_max), and outer engine cutoff time (T_outer_engine_cutoff_min to T_outer_engine_cutoff_max)
bounds = [(0, 1)] * N + [(T_burn_min, T_burn_max), (T_outer_engine_cutoff_min, T_outer_engine_cutoff_max)]

constraints = [
    {'type': 'ineq', 'fun': constr_final_y},
    {'type': 'ineq', 'fun': constr_final_v},
    {'type': 'ineq', 'fun': constr_dynamic},
    {'type': 'ineq', 'fun': constr_propellant}  # Add propellant mass constraint
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
        'gtol': 4e-1,
        'xtol': 4e-1,
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