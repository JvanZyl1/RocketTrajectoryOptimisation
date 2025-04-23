import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Updated model parameters from user
T_e = 2745000.0   # engine thrust constant [N]
n_e = 9.0           # thrust efficiency
v_ex = 3433.5       # exhaust velocity [m/s]
g0 = 9.80665        # gravity [m/s^2]
T_base = 288.15     # ISA base temperature [K]
p_base = 101325.0   # ISA base pressure [Pa]
a_lapse = -0.0065   # temperature lapse rate [K/m]
R = 287.05          # specific gas constant [J/(kg·K)]
Q_max = 30000.0     # max dynamic pressure [Pa]

# Discretisation
N = 500             # number of time intervals
T_end_min = 10.0    # minimum burn time [s]
T_end_max = 60.0    # maximum burn time [s]

# Updated initial conditions
y0 = 4985.673145885028    # initial altitude [m]
vy0 = -145.2026108716095  # initial vertical velocity [m/s]
m0 = 1025137.8421162822   # initial mass [kg]
m_strc = 368012.9065218879 # structure mass [kg]
mp0 = m0 - m_strc          # propellant mass [kg]

# Density as function of altitude
def density(y):
    T = T_base + a_lapse * y
    p = p_base * (T / T_base) ** (g0 / (a_lapse * R))
    return p / (R * T)

# Simulate states and dynamic-pressure margins
def simulate(u, T_end):
    dt = T_end / N
    y = y0
    vy = vy0
    m = m0
    mp = mp0  # initial propellant mass
    ys, vys, margins, mps = [], [], [], []  # track propellant mass
    for ui in u:
        rho_val = density(y)
        margin = Q_max - 0.5 * rho_val * vy**2
        margins.append(margin)
        mdot = -ui * T_e * n_e / v_ex
        vy_dot = -g0 + ui * T_e * n_e / m
        y += vy * dt
        vy += vy_dot * dt
        m += mdot * dt
        mp += mdot * dt  # update propellant mass
        ys.append(y)
        vys.append(vy)
        mps.append(mp)
    return np.array(ys), np.array(vys), m, np.array(margins), np.array(mps)

# Objective: minimise propellant used
def objective(x):
    u = x[:-1]  # throttle values
    T_end = x[-1]  # burn time
    sim = simulate(u, T_end)
    return m0 - sim[2]

# Constraints
def constr_final_y(x):
    u = x[:-1]
    T_end = x[-1]
    y_final = simulate(u, T_end)[0][-1]
    return np.array([y_final - 4.75, 5.0 - y_final])  # y_final must be between 9.5 and 10.5

def constr_final_v(x):
    u = x[:-1]
    T_end = x[-1]
    v_final = simulate(u, T_end)[1][-1]
    return np.array([0.05 - v_final, 0.05 + v_final])  # |v_final| must be less than 0.5

def constr_dynamic(x):
    u = x[:-1]
    T_end = x[-1]
    return simulate(u, T_end)[3]  # must be >= 0

def constr_propellant(x):
    u = x[:-1]
    T_end = x[-1]
    mps = simulate(u, T_end)[4]  # get propellant mass history
    return mps  # must be >= 0

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

    def __call__(self, xk, state=None):
        u = xk[:-1]
        T_end = xk[-1]
        # Calculate constraint violations
        y_final = simulate(u, T_end)[0][-1]
        v_final = simulate(u, T_end)[1][-1]
        mps = simulate(u, T_end)[4]
        min_mp = np.min(mps)
        
        y_violation = max(0, y_final - 9.5, 10.5 - y_final)
        v_violation = max(0, abs(v_final) - 0.5)
        dynamic_violations = np.minimum(constr_dynamic(xk), 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        prop_violation = max(0, -min_mp)
        
        # Calculate objective value
        obj_val = objective(xk)
        
        # Store data
        self.iterations.append(len(self.iterations))
        self.objectives.append(obj_val)
        self.alt_violations.append(y_violation)
        self.vel_violations.append(v_violation)
        self.dyn_violations.append(max_dynamic_violation)
        self.prop_violations.append(prop_violation)
        self.burn_times.append(T_end)
        self.final_alts.append(y_final)
        self.final_vels.append(v_final)
        self.min_props.append(min_mp)
        
        print(f"\nIteration Information:")
        print(f"Burn time: {T_end:.2f} s")
        print(f"Final altitude: {y_final:.3f} m")
        print(f"Final velocity: {v_final:.3f} m/s")
        print(f"Minimum propellant mass: {min_mp:.2f} kg")
        print(f"Objective value: {obj_val:.2f} kg")
        print(f"Final altitude constraint violation: {y_violation:.3e} m")
        print(f"Final velocity constraint violation: {v_violation:.3e} m/s")
        print(f"Maximum dynamic pressure constraint violation: {max_dynamic_violation:.2e} Pa")
        print(f"Propellant mass constraint violation: {prop_violation:.2e} kg")
        return False

def plot_optimization_history(callback):
    plt.figure(figsize=(15, 10))
    
    # Plot objective and constraints
    plt.subplot(2, 2, 1)
    plt.plot(callback.iterations, callback.objectives, 'b-', label='Objective')
    plt.xlabel('Iteration')
    plt.ylabel('Propellant Used [kg]')
    plt.title('Objective Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(callback.iterations, callback.alt_violations, 'g-', label='Altitude')
    plt.semilogy(callback.iterations, callback.vel_violations, 'r-', label='Velocity')
    plt.semilogy(callback.iterations, callback.dyn_violations, 'b-', label='Dynamic Pressure')
    plt.semilogy(callback.iterations, callback.prop_violations, 'k-', label='Propellant')
    plt.xlabel('Iteration')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(callback.iterations, callback.burn_times, 'm-')
    plt.xlabel('Iteration')
    plt.ylabel('Burn Time [s]')
    plt.title('Burn Time')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(callback.iterations, callback.final_alts, 'g-', label='Altitude')
    plt.plot(callback.iterations, callback.final_vels, 'r-', label='Velocity')
    plt.xlabel('Iteration')
    plt.ylabel('Final State')
    plt.title('Final Altitude and Velocity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_history.png')
    plt.close()

def plot_simulation_results(u_opt, T_end_opt):
    ys, vys, m_final, margins, mps = simulate(u_opt, T_end_opt)
    t = np.linspace(0, T_end_opt, N)
    
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
    plt.plot(t, u_opt, 'g-')
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
    plt.savefig('simulation_results.png')
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

def save_final_parameters(u_opt, T_end_opt, filename='final_parameters.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Burn_Time', T_end_opt])
        writer.writerow(['Throttle_Profile'] + list(u_opt))

# Initial guess, bounds, assembly
# Create a more realistic initial guess
u0 = np.zeros(N)
# Start with full throttle to slow down
u0[:N//3] = 1.0
# Gradually reduce throttle
u0[N//3:2*N//3] = 0.7
# Final approach with lower throttle
u0[2*N//3:] = 0.3

T_end0 = 15.0  # initial guess for burn time
x0 = np.append(u0, T_end0)  # combine throttle and burn time into single vector

# Bounds for throttle (0 to 1) and burn time (T_end_min to T_end_max)
bounds = [(0, 1)] * N + [(T_end_min, T_end_max)]

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
        'maxiter': 1000,
        'verbose': 2,
        'gtol': 1e-6,
        'xtol': 1e-2,
        'barrier_tol': 1e-6,
        'initial_tr_radius': 1.0
    },
    callback=callback
)

# Save and plot results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_optimization_data(callback, f'optimization_history_{timestamp}.csv')
plot_optimization_history(callback)

# Output results
if result.success:
    x_opt = result.x
    u_opt = x_opt[:-1]
    T_end_opt = x_opt[-1]
    save_final_parameters(u_opt, T_end_opt, f'final_parameters_{timestamp}.csv')
    plot_simulation_results(u_opt, T_end_opt)
    
    print("\nOptimisation converged")
    print(f"Optimal burn time: {T_end_opt:.2f} s")
    print(f"Initial mass: {m0:.2f} kg")
    print(f"Final mass:   {m_final:.2f} kg")
    print(f"Final altitude: {ys[-1]:.3f} m")
    print(f"Final velocity: {vys[-1]:.3f} m/s")
    print("Throttle profile (first 10 values):")
    print(u_opt[:10])
else:
    print("Optimisation failed:", result.message)
    # Print final state even if optimization failed
    x_final = result.x
    u_final = x_final[:-1]
    T_end_final = x_final[-1]
    ys, vys, m_final, _, _ = simulate(u_final, T_end_final)
    print(f"\nFinal state at failure:")
    print(f"Burn time: {T_end_final:.2f} s")
    print(f"Final altitude: {ys[-1]:.3f} m")
    print(f"Final velocity: {vys[-1]:.3f} m/s")
    print(f"Final mass: {m_final:.2f} kg")
