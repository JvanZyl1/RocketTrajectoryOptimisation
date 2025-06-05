# -----------------------------------------------------------------------------
# PACKAGES
# -----------------------------------------------------------------------------
import csv
import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

# -----------------------------------------------------------------------------
# INITIAL STATE
# -----------------------------------------------------------------------------
state_initial = load_landing_burn_initial_state()
y_0 = state_initial[1]
v_0 = state_initial[3]   # negative = downward
m_0 = state_initial[8]   # structural; add propellant if desired

print(f'Initial mass: {m_0:.1f} kg, v0: {v_0:.1f} m/s, y0: {y_0:.1f} m')

# -----------------------------------------------------------------------------
# VEHICLE PARAMETERS
# -----------------------------------------------------------------------------
sizing_results: dict[str, str] = {}
with open('data/rocket_parameters/sizing_results.csv', newline='') as fh:
    for key, *_, value in csv.reader(fh):
        sizing_results[key] = value

Te = float(sizing_results['Thrust engine stage 1'])
T = Te * int(sizing_results['Number of engines gimballed stage 1'])
mdot = T / float(sizing_results['Exhaust velocity stage 1'])
C_n_0 = float(sizing_results['C_n_0'])
S_grid_fins = float(sizing_results['S_grid_fins'])
n_gf = 4
m_s = float(sizing_results['Structural mass stage 1 (descent)']) * 1_000

print(f'Initial propellant mass: {m_0 - m_s:.1f} kg')
number_of_engines_min = 3
minimum_engine_throttle = 0.4
#tau_min = (
#    number_of_engines_min * minimum_engine_throttle
#) / int(sizing_results['Number of engines gimballed stage 1'])
#assert 0 < tau_min < 1, 'tau_min outside (0,1). Check sizing results.'
tau_min = 0.0
# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
g_0 = 9.80665
q_max = 40_000  # Pa

# -----------------------------------------------------------------------------
# REFERENCE TRAJECTORY (FOR INITIAL GUESS)
# -----------------------------------------------------------------------------
ref = pd.read_csv(
    'data/reference_trajectory/landing_burn_controls/reference_trajectory_landing_burn_control.csv'
)

t_ref = ref['t[s]'].to_numpy()
y_ref = ref['y[m]'].to_numpy()
v_ref = ref['vy[m/s]'].to_numpy()
m_ref = ref['mass[kg]'].to_numpy()
tau_ref = ref['tau[-]'].to_numpy()

# -----------------------------------------------------------------------------
# ATMOSPHERE DENSITY (5th‑order poly fit)
# -----------------------------------------------------------------------------
y_grid = np.linspace(0, 50_000, 500)
rho_vals = np.array([endo_atmospheric_model(h)[0] for h in y_grid])
coeffs = np.polyfit(y_grid, rho_vals, 5)

def rho_fun(h):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = p * h + c
    return p

# -----------------------------------------------------------------------------
# OPTIMISATION SET‑UP (multiple shooting)
# -----------------------------------------------------------------------------
N = 2000                      # mesh segments
opti = ca.Opti()

# Scaled decision variables
y_hat  = opti.variable(N + 1)     # altitude / 40_000  
v_hat  = opti.variable(N + 1)     # vertical velocity / 1_700  
m_hat  = opti.variable(N + 1)     # (mass − m_s) / 7e5  
t_hatf = opti.variable()          # burn time / 50
eta = opti.variable(N)        # unconstrained throttle parameter

dt = t_hatf * 50 / N

# tanh mapping keeps τ in [tau_min, 1] without overflow
tau = tau_min + 0.5 * (1.0 - tau_min) * (1 + ca.tanh(eta))

# -----------------------------------------------------------------------------
# CONSTRAINTS
# -----------------------------------------------------------------------------
# Initial conditions
opti.subject_to([y_hat[0] == y_0/40_000, v_hat[0] == v_0/1_700, m_hat[0] == (m_0 - m_s)/7e5])

# Terminal box
opti.subject_to(ca.fabs(y_hat[-1] * 40_000) <= 5.0)  # Relaxed from 2.0
opti.subject_to(ca.fabs(v_hat[-1] * 1_700) <= 2.0)  # Relaxed from 1.0
opti.subject_to(m_hat[-1] * 7e5 + m_s >= m_s)

# Throttle lower bound (upper bound implicit)
opti.subject_to(tau >= tau_min)

# Allow solver to stretch the burn
opti.subject_to(t_hatf * 50 >= 50.0)

# Dynamics and path constraints
for k in range(N):
    y_k = y_hat[k] * 40_000
    v_k = v_hat[k] * 1_700
    m_k = m_hat[k] * 9e6
    
    q_k = 0.5 * rho_fun(y_k) * v_k ** 2
    a_k = T / m_k * tau[k] - g_0 + q_k * C_n_0 * S_grid_fins * n_gf / m_k

    opti.subject_to(y_hat[k + 1] == (y_k + dt * v_k) / 40_000)
    opti.subject_to(v_hat[k + 1] == (v_k + dt * a_k) / 1_700)
    opti.subject_to(m_hat[k + 1] == (m_k - dt * mdot * tau[k]) / 9e6)

    if k % 4 == 0:
        opti.subject_to(q_k <= q_max)
    opti.subject_to(v_k <= 0.1)      # relaxed downward constraint

# Objective: maximise final mass
opti.minimize(-(m_hat[-1] * 7e5 + m_s))

# -----------------------------------------------------------------------------
# INITIAL GUESS
# -----------------------------------------------------------------------------
opti.set_initial(t_hatf, max(1.0, t_ref[-1]/50))

time_mesh = np.linspace(0, t_ref[-1], N + 1)

opti.set_initial(y_hat, np.interp(time_mesh, t_ref, y_ref)/40_000)
opti.set_initial(v_hat, np.interp(time_mesh, t_ref, v_ref)/1_700)
opti.set_initial(m_hat, (np.interp(time_mesh, t_ref, m_ref) - m_s)/7e5)

# throttle → eta via arctanh
raw_tau = np.interp(time_mesh[:-1], t_ref, tau_ref)
clipped = np.clip(raw_tau, tau_min + 1e-3, 1.0 - 1e-3)
scale   = 2 * (clipped - tau_min) / (1.0 - tau_min) - 1  # in (‑1, 1)
scale   = np.clip(scale, -0.999, 0.999)
eta_init = np.arctanh(scale)

opti.set_initial(eta, eta_init)

# -----------------------------------------------------------------------------
# SOLVER OPTIONS
# -----------------------------------------------------------------------------
# Enhanced solver settings for improved numerical conditioning and convergence
p_opts = {"expand": True}
s_opts = {
    "linear_solver":     "mumps",
    "mumps_mem_percent": 3000,
    "hessian_approximation": "limited-memory",
    "max_iter":          600,
    "tol":               1e-3,
    "warm_start_init_point": "yes",
    "print_level":       5,
    "print_frequency_iter": 5,        # Print every 10 iterations
    "print_timing_statistics": "yes",  # Show timing information
    "print_user_options": "yes"        # Show user options
}

opti.solver("ipopt", p_opts, s_opts)

# -----------------------------------------------------------------------------
# SOLVE
# -----------------------------------------------------------------------------
print("Starting optimisation …")
try:
    sol = opti.solve()
    print("Optimal solution found")
    solved = True
except RuntimeError as err:
    print(f"Solver failed: {err}")
    solved = False

# -----------------------------------------------------------------------------
# PLOT RESULTS
# -----------------------------------------------------------------------------
if not solved:
    print("\n--- last-iterate box residuals ---")
    print("y_end  :", opti.debug.value(y_hat[-1]) * 40_000)
    print("v_end  :", opti.debug.value(v_hat[-1]) * 1_700)
    print("m_prop end  :", opti.debug.value(m_hat[-1]) * 7e5, "(> 0 means OK)")
    print("max |q|:", np.max(0.5 *
                            rho_fun(opti.debug.value(y_hat)[:-1] * 40_000) *
                            opti.debug.value(v_hat)[:-1]**2 * 1_700**2))
    print("min τ  :", np.min(opti.debug.value(tau)), " (should be >= ", tau_min, ")")
    print("max τ  :", np.max(opti.debug.value(tau)), " (should be 1)")

    # Get debug values
    y_debug = opti.debug.value(y_hat) * 40_000
    v_debug = opti.debug.value(v_hat) * 1_700
    m_debug = opti.debug.value(m_hat) * 7e5 + m_s
    tau_debug = opti.debug.value(tau)
    t_f_debug = opti.debug.value(t_hatf) * 50
    t_grid = np.linspace(0, t_f_debug, N + 1)

    # Create debug plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # Altitude
    axs[0].plot(t_grid, y_debug)
    axs[0].set_ylabel('Altitude [m]')
    axs[0].grid(True)
    axs[0].axhline(y=2.0, color='r', linestyle='--', label='Terminal bound')
    axs[0].axhline(y=-2.0, color='r', linestyle='--')
    axs[0].legend()

    # Velocity
    axs[1].plot(t_grid, v_debug)
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].grid(True)
    axs[1].axhline(y=0.1, color='r', linestyle='--', label='Downward only')
    axs[1].axhline(y=2.0, color='g', linestyle='--', label='Terminal bound')
    axs[1].axhline(y=-2.0, color='g', linestyle='--')
    axs[1].legend()

    # Throttle
    axs[2].step(t_grid[:-1], tau_debug, where='post')
    axs[2].set_ylabel('Throttle')
    axs[2].grid(True)
    axs[2].axhline(y=tau_min, color='r', linestyle='--', label='Min throttle')
    axs[2].axhline(y=1.0, color='g', linestyle='--', label='Max throttle')
    axs[2].legend()

    # Dynamic pressure
    q_debug = 0.5 * rho_fun(y_debug[:-1]) * v_debug[:-1]**2
    axs[3].plot(t_grid[:-1], q_debug/1000)  # Convert to kPa
    axs[3].set_ylabel('Dynamic Pressure [kPa]')
    axs[3].set_xlabel('Time [s]')
    axs[3].grid(True)
    axs[3].axhline(y=q_max/1000, color='r', linestyle='--', label='Max q')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig('debug_trajectory.png')
    plt.show()