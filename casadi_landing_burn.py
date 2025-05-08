import csv
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
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
n_e = 12 # select yourself.
q_max = 30000 # select yourself.
T = Te * n_e
m_s = float(sizing_results['Structural mass stage 1 (descent)'])*1000
v_ex = float(sizing_results['Exhaust velocity stage 1'])
mdot = T / v_ex
C_n_0 = 2#float(sizing_results['C_n_0'])
S_grid_fins = 4#float(sizing_results['S_grid_fins'])

g_0 = 9.80665

# ---- SETUP ----
N = 8000        # Discretisation steps
opti = ca.Opti()

y = opti.variable(N+1)
v = opti.variable(N+1)
m = opti.variable(N+1)
tau = opti.variable(N)
t_f = opti.variable()
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


# a linear guess from start→end
opti.set_initial(y, np.linspace(y_0, 0, N+1))
opti.set_initial(v, np.linspace(v_0, 0, N+1))
opti.set_initial(m, np.linspace(m_0, m_s, N+1))
opti.set_initial(tau, 0.5)
opti.set_initial(t_f, y_0 / abs(v_0) * 1.5)


for k in range(N):
    a = T / m[k] * tau[k] - g_0 + 0.5 * rho_fun(y[k]) * v[k]**2 * C_n_0 * S_grid_fins
    opti.subject_to(y[k+1] == y[k] + dt * v[k])
    opti.subject_to(v[k+1] == v[k] + dt * a)
    opti.subject_to(m[k+1] == m[k] - dt * mdot * tau[k])
    opti.subject_to(m[k] >= m_s)
    opti.subject_to(v[k] <= v_max(y[k]))
    opti.subject_to(tau[k] >= 0)
    opti.subject_to(tau[k] <= 1)

opti.minimize(-m[-1])  # Maximize final mass

# Set IPOPT options for progress display
p_opts = {"expand": True}
s_opts = {
    "max_iter": 1000,
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

    # boundary residuals
    r_y_end = y_guess[-1]
    r_v_end = v_guess[-1]
    print(f"y_end residual: {r_y_end:.3e}")
    print(f"v_end residual: {r_v_end:.3e}\n")

    # dynamics defects
    a_k    = T/m_guess[:-1]*tau_guess - g_0 \
             + 0.5 * rho_fun(y_guess[:-1]) * v_guess[:-1]**2 * C_n_0 * S_grid_fins
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

    t_grid = np.linspace(0, t_f_guess, N+1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t_grid, y_guess)
    axs[0].set_ylabel("Altitude [m]")
    axs[1].plot(t_grid, v_guess)
    axs[1].set_ylabel("Velocity [m/s]")
    axs[2].step(t_grid[:-1], tau_guess, where='post')
    axs[2].set_ylabel("Throttle")
    axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()