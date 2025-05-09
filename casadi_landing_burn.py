import csv
import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.base_environment import load_landing_burn_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model


state_initial = load_landing_burn_initial_state()
y_0 = state_initial[1]
v_0 = state_initial[3] # approx equal to vy_0
m_0 = state_initial[8] + 2000000

# ---- PARAMETERS (to be filled in by user) ----
sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]

Te = float(sizing_results['Thrust engine stage 1'])
n_e = int(sizing_results['Number of engines gimballed stage 1'])
q_max = 32000 # select yourself.
T = Te * n_e
m_s = float(sizing_results['Structural mass stage 1 (descent)'])*1000
v_ex = float(sizing_results['Exhaust velocity stage 1'])
mdot = T / v_ex
C_n_0 = float(sizing_results['C_n_0'])
S_grid_fins = float(sizing_results['S_grid_fins'])
n_gf = 4

number_of_engines_min = 3
minimum_engine_throttle = 0.4
tau_min = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])


g_0 = 9.80665

# ---- REFERENCE TRAJECTORY ----
df_reference = pd.read_csv('data/reference_trajectory/landing_burn_optimal/reference_trajectory_landing_burn_control.csv')
data_t = df_reference['t[s]'].values
data_y = df_reference['y[m]'].values
data_v = df_reference['vy[m/s]'].values
data_m = df_reference['mass[kg]'].values
data_tau = df_reference['tau[-]'].values

# ---- SETUP ----
N = int((data_t[-1] + 20 - data_t[0])/0.01)
opti = ca.Opti()

y = opti.variable(N+1)
v = opti.variable(N+1)
m = opti.variable(N+1)
tau = opti.variable(N)
t_f = opti.variable()
opti.set_initial(t_f, data_t[-1])
opti.subject_to(t_f >= 1e-2)      # final time must be positive
dt  = t_f / N                     # uniform step length


# ----ATMOSPHERE DYNAMICS----
# Sample ISA density
y_grid = np.linspace(0, 45000, 500)
rho_vals = np.array([endo_atmospheric_model(y)[0] for y in y_grid])
# Fit a 5th-degree poly: ρ(y) = c0 + c1·y + c2·y^2 + … + c5·y^5
coeffs = np.polyfit(y_grid, rho_vals, 5)  
# coeffs[0]·y^5 + coeffs[1]·y^4 + … + coeffs[5]
print("rho_poly_coeffs =", coeffs)

'''
# check fit with plot
plt.figure()
plt.plot(y_grid, rho_vals, 'b-', label='Original Data')
plt.plot(y_grid, np.polyval(coeffs, y_grid), 'r-', label='Fitted Poly')
plt.legend()
plt.show()
'''

rho_poly_coeffs = [coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]]

# Build a CasADi-compatible density function
def rho_fun(y):
    # Horner's method for poly evaluation
    p = rho_poly_coeffs[0]
    for ci in rho_poly_coeffs[1:]:
        p = p * y + ci
    return p

# Max dynamic-pressure velocity
q_max = 30000  # Pa
v_max = lambda y: ca.sqrt(2 * q_max / rho_fun(y))

opti.subject_to(y[0] == y_0)
opti.subject_to(v[0] == v_0)
opti.subject_to(m[0] == m_0)
tol_y = 3
tol_v = 0.5
opti.subject_to(y[-1] <=  tol_y)
opti.subject_to(y[-1] >= -tol_y)
opti.subject_to(v[-1] <=  tol_v)
opti.subject_to(v[-1] >= -tol_v)


# a linear guess from start-> end
# load initial guesses


# linear interpolation
opti.set_initial(t_f, data_t[-1])
y_guess = np.interp(np.linspace(0, data_t[-1], N+1), data_t, data_y)
v_guess = np.interp(np.linspace(0, data_t[-1], N+1), data_t, data_v)
m_guess = np.interp(np.linspace(0, data_t[-1], N+1), data_t, data_m)
tau_guess = np.interp(np.linspace(0, data_t[-1], N), data_t, data_tau)

opti.set_initial(y, y_guess)
opti.set_initial(v, v_guess)
opti.set_initial(m, m_guess)
opti.set_initial(tau, tau_guess)


for k in range(N):
    a = T / m[k] * tau[k] - g_0 + 0.5 * rho_fun(y[k]) * v[k]**2 * C_n_0 * S_grid_fins * n_gf / m[k]
    opti.subject_to(y[k+1] == y[k] + dt * v[k])
    opti.subject_to(v[k+1] == v[k] + dt * a)
    opti.subject_to(m[k+1] == m[k] - dt * mdot * tau[k])
    opti.subject_to(m[k] >= m_s)
    opti.subject_to(v[k] <= v_max(y[k]))
    opti.subject_to(tau[k] >= tau_min)
    opti.subject_to(tau[k] <= 1)

opti.minimize(-m[-1])  # Maximize final mass

# Set IPOPT options for progress display
p_opts = {"expand": True}
s_opts = {
    "max_iter": 200,
    "print_level": 5,     # 0-12, higher means more output
    "print_frequency_iter": 10,  # Print every 10 iterations
    "print_timing_statistics": "yes"
}

print("Starting optimization...")
# Solve the NLP and only proceed on success
opti.solver('ipopt', p_opts, s_opts)

try:
    sol = opti.solve()
    solved = True
except RuntimeError as e:
    solved = False
    print(f"Solver failed: {e}")

if solved:
    y_opt   = sol.value(y)
    v_opt   = sol.value(v)
    m_opt   = sol.value(m)
    tau_opt = sol.value(tau)
    t_f_opt = sol.value(t_f)
    dt_opt  = sol.value(dt)
    t_grid  = np.linspace(0, t_f_opt, N+1)
    
    # ---- PLOT ----
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t_grid, y_opt)
    axs[0].set_ylabel("Altitude [m]")
    axs[1].plot(t_grid, v_opt)
    axs[1].set_ylabel("Velocity [m/s]")
    axs[2].step(t_grid[:-1], tau_opt, where='post')
    axs[2].set_ylabel("Throttle")
    axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    pass
else:
    # retrieve last iterate
    y_guess   = opti.debug.value(y)
    v_guess   = opti.debug.value(v)
    m_guess   = opti.debug.value(m)
    tau_guess = opti.debug.value(tau)
    t_f_guess = opti.debug.value(t_f)
    dt_guess  = t_f_guess / N
    dynamic_pressure_guess = 0.5 * rho_fun(y_guess[:-1]) * v_guess[:-1]**2
    acs_force_guess = C_n_0 * S_grid_fins * dynamic_pressure_guess * n_gf

    # boundary residuals
    r_y_end = y_guess[-1]
    r_v_end = v_guess[-1]
    print(f"y_end residual: {r_y_end:.3e}")
    print(f"v_end residual: {r_v_end:.3e}\n")

    # dynamics defects
    a_k    = T/m_guess[:-1]*tau_guess - g_0 \
             + 0.5 * rho_fun(y_guess[:-1]) * v_guess[:-1]**2 * C_n_0 * S_grid_fins * n_gf / m_guess[:-1]
    dy_res = y_guess[1:] - (y_guess[:-1] + dt_guess*v_guess[:-1])
    dv_res = v_guess[1:] - (v_guess[:-1] + dt_guess*a_k)
    print(f"max |delta y| = {np.max(np.abs(dy_res)):.3e}")
    print(f"max |delta v| = {np.max(np.abs(dv_res)):.3e}\n")

    # path constraints
    vlim = opti.debug.value(v_max(y_guess[:-1]))
    viol = v_guess[:-1] - vlim
    print(f"max (v-v_max) = {np.max(viol):.3e}\n")

    # mass constraint
    m_viol = m_s - m_guess
    print(f"max (m_s-m) = {np.max(m_viol):.3e}")
    
    # Print all constraint violations in one place for easier analysis
    print("\n----- CONSTRAINT VIOLATION SUMMARY -----")
    # Initial conditions
    print(f"Initial y constraint: {abs(y_guess[0] - y_0):.3e}")
    print(f"Initial v constraint: {abs(v_guess[0] - v_0):.3e}")
    print(f"Initial m constraint: {abs(m_guess[0] - m_0):.3e}")
    
    # Final conditions
    print(f"Final y upper bound: {max(0, y_guess[-1] - tol_y):.3e}")
    print(f"Final y lower bound: {max(0, -tol_y - y_guess[-1]):.3e}")
    print(f"Final v upper bound: {max(0, v_guess[-1] - tol_v):.3e}")
    print(f"Final v lower bound: {max(0, -tol_v - v_guess[-1]):.3e}")
    
    # Dynamics constraints (max violations)
    print(f"Altitude dynamics: {np.max(np.abs(dy_res)):.3e}")
    print(f"Velocity dynamics: {np.max(np.abs(dv_res)):.3e}")
    
    # Path constraints (max violations)
    print(f"Mass lower bound: {max(0, np.max(m_viol)):.3e}")
    print(f"Velocity upper bound: {max(0, np.max(viol)):.3e}")
    
    # Check throttle constraints
    tau_min_viol = -np.min(tau_guess)
    tau_max_viol = np.max(tau_guess) - 1
    print(f"Throttle lower bound: {max(0, tau_min_viol):.3e}")
    print(f"Throttle upper bound: {max(0, tau_max_viol):.3e}")
    
    # Check t_f constraint
    print(f"Final time constraint: {max(0, 1e-2 - t_f_guess):.3e}")
    print("--------------------------------------")

    t_grid = np.linspace(0, t_f_guess, N+1)
    plt.figure(figsize=(20,15))
    plt.suptitle("Landing Burn Optimization (constraints violated)", fontsize = 20)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1], width_ratios=[1,1], hspace = 0.3, wspace = 0.3)
    ax1 = plt.subplot(gs[0,0])
    ax1.plot(t_grid, y_guess, color = 'blue', linewidth = 4)
    ax1.set_ylabel("Altitude [m]", fontsize = 20)
    ax1.set_xlabel("Time [s]", fontsize = 20)
    ax1.set_title("Altitude Profile", fontsize = 20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True)
    ax2 = plt.subplot(gs[0,1])
    ax2.plot(t_grid, v_guess, color = 'blue', linewidth = 4)
    ax2.set_ylabel("Velocity [m/s]", fontsize = 20)
    ax2.set_xlabel("Time [s]", fontsize = 20)
    ax2.set_title("Velocity Profile", fontsize = 20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True)
    ax3 = plt.subplot(gs[1,0])
    ax3.step(t_grid[:-1], tau_guess, where='post', color = 'blue', linewidth = 4)
    ax3.set_ylabel("Throttle", fontsize = 20)
    ax3.set_xlabel("Time [s]", fontsize = 20)
    ax3.set_title("Throttle Profile", fontsize = 20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.grid(True)
    ax4 = plt.subplot(gs[1,1])
    ax4.plot(t_grid, m_guess, color = 'blue', linewidth = 4)
    ax4.set_ylabel("Mass [kg]", fontsize = 20)
    ax4.set_xlabel("Time [s]", fontsize = 20)
    ax4.set_title("Mass Profile", fontsize = 20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)
    ax5 = plt.subplot(gs[2,0])
    if max(acs_force_guess) > 1e6:
        ax5.plot(t_grid[:-1], acs_force_guess/1e6, color = 'blue', linewidth = 4)
        ax5.set_ylabel("ACS Force [MN]", fontsize = 20)
    elif max(acs_force_guess) > 1e3:
        ax5.plot(t_grid[:-1], acs_force_guess/1e3, color = 'blue', linewidth = 4)
        ax5.set_ylabel("ACS Force [kN]", fontsize = 20)
    else:
        ax5.plot(t_grid[:-1], acs_force_guess, color = 'blue', linewidth = 4)
        ax5.set_ylabel("ACS Force [N]", fontsize = 20)
    ax5.set_xlabel("Time [s]", fontsize = 20)
    ax5.set_title("ACS Force Profile", fontsize = 20)
    ax5.tick_params(axis='both', labelsize=16)
    ax5.grid(True)
    ax6 = plt.subplot(gs[2,1])
    ax6.plot(t_grid, dynamic_pressure_guess/1000, color = 'blue', linewidth = 4)
    ax6.set_ylabel("Dynamic Pressure [kPa]", fontsize = 20)
    ax6.set_xlabel("Time [s]", fontsize = 20)
    ax6.set_title("Dynamic Pressure Profile", fontsize = 20)
    ax6.tick_params(axis='both', labelsize=16)
    ax6.grid(True)
    plt.savefig('results/landing_burn_optimal/optimal_landing_burn.png')
    plt.show()