import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import csv
import dill
from src.envs.utils.reference_trajectory_interpolation import get_dt
from src.TrajectoryGeneration.Transformations import calculate_flight_path_angles

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD

from src.envs.rockets_physics import compile_physics


# Load data from data/reference_trajectory/reference_trajectory_endo_clean.csv
data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')

# Data is t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
times_r = data['t[s]'].values
x_r = data['x[m]'].values
y_r = data['y[m]'].values
vx_r = data['vx[m/s]'].values
vy_r = data['vy[m/s]'].values
mass_r = data['mass[kg]'].values

# Calculate flight path angles
gamma_r = np.deg2rad(calculate_flight_path_angles(vx_r, vy_r))

# step 1 interpolate to have a constant time step
dt = get_dt()
times_r_interp = np.arange(times_r[0], times_r[-1], dt)
x_r_interp = interp.interp1d(times_r, x_r)(times_r_interp)
y_r_interp = interp.interp1d(times_r, y_r)(times_r_interp)
vx_r_interp = np.zeros_like(x_r_interp)
vy_r_interp = np.zeros_like(y_r_interp)

for k in range(len(times_r_interp) - 1):
    vx_r_interp[k+1] = (x_r_interp[k+1] - x_r_interp[k]) / dt
    vy_r_interp[k+1] = (y_r_interp[k+1] - y_r_interp[k]) / dt

masses_r_interp = interp.interp1d(times_r, mass_r)(times_r_interp)
gamma_r_interp = np.deg2rad(calculate_flight_path_angles(vy_r_interp, vx_r_interp))
gamma_r_interp[0] = math.radians(90)

Moments_z = np.zeros(len(times_r_interp) - 1)
Thrust_parallel = np.zeros(len(times_r_interp) - 1)
Thrust_perpendicular = np.zeros(len(times_r_interp) - 1)

masses = np.zeros(len(times_r_interp))
masses[0] = mass_r[0]


kl_sub = 2.0
kl_sup = 1.0
cd0_subsonic=0.05
kd_subsonic=0.5
cd0_supersonic=0.10
kd_supersonic=1.0
CL_func = lambda alpha, M: rocket_CL(alpha, M, kl_sub, kl_sup)
CD_func = lambda alpha, M: rocket_CD(alpha, M, cd0_subsonic, kd_subsonic, cd0_supersonic, kd_supersonic)

sizing_results = {}
with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]

with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
    rocket_functions = dill.load(f)

frontal_area = float(sizing_results['Rocket frontal area'])
T_e_nolosses = float(sizing_results['Thrust engine stage 1'])
p_e = float(sizing_results['Nozzle exit pressure stage 1'])
A_e = float(sizing_results['Nozzle exit area'])
v_ex = float(sizing_results['Exhaust velocity stage 1'])
m_p_0 = (float(sizing_results['Propellant mass stage 1 (ascent)']) \
         + float(sizing_results['Propellant mass stage 1 (descent)']))*1000
cop_func = rocket_functions['cop_subrocket_0_lambda']
cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda']
maximum_Mz_moment = 0.75e9
maximum_F_parallel_thrust = 1.1e8
maximum_F_perpendicular_thrust = 1.75e7
minimum_F_parallel_thrust_factor = 0.7
m_p = np.zeros(len(times_r_interp))
m_p[0] = m_p_0

actions = np.zeros((len(times_r_interp) - 1, 3))


vx_sims = np.zeros_like(times_r_interp)
vx_sims[0] = 0
vy_sims = np.zeros_like(times_r_interp)
vy_sims[0] = 0

x_sims = np.zeros_like(times_r_interp)
y_sims = np.zeros_like(times_r_interp)
x_sims[0] = 0
y_sims[0] = 0

masses_sims = np.zeros_like(times_r_interp)
masses_sims[0] = masses_r_interp[0]

gamma_deg_sims = np.zeros_like(times_r_interp)
gamma_deg_sims[0] = 90

for k in range(len(times_r_interp) - 1):
    # Calculating independent to action variables.
    g_k = gravity_model_endo(y_r_interp[k])
    rho_k, p_k, a_k = endo_atmospheric_model(y_r_interp[k])

    speed_k = np.sqrt(vx_r_interp[k]**2 + vy_r_interp[k]**2)
    mach_k = speed_k / a_k
    C_D_k = CD_func(0, mach_k)
    C_L_k = CL_func(0, mach_k)
    cop_k = cop_func(0, mach_k)

    L_k = 0.5 * rho_k * speed_k**2 * C_L_k * frontal_area
    D_k = 0.5 * rho_k * speed_k**2 * C_D_k * frontal_area

    F_x_aero_k = - D_k * np.cos(gamma_r_interp[k]) - L_k * np.cos(math.pi - gamma_r_interp[k])
    F_y_aero_k = - D_k * np.sin(gamma_r_interp[k]) + L_k * np.sin(math.pi - gamma_r_interp[k])

    T_e_losses_k = T_e_nolosses + (p_e - p_k) * A_e

    # Acceleration required.
    vdot_x_k = (vx_r_interp[k+1] - vx_r_interp[k]) / dt
    vdot_y_k = (vy_r_interp[k+1] - vy_r_interp[k]) / dt

    # Update simulation storage for debugging
    vx_sims[k+1] = vx_sims[k] + vdot_x_k * dt
    vy_sims[k+1] = vy_sims[k] + vdot_y_k * dt
    x_sims[k+1] = x_sims[k] + vx_sims[k+1] * dt
    y_sims[k+1] = y_sims[k] + vy_sims[k+1] * dt
    gamma_deg_sims[k+1] = np.rad2deg(np.arctan2(vy_sims[k+1], vx_sims[k+1]))

    # Forces required.
    F_x_k = vdot_x_k * masses[k]
    F_y_k = (vdot_y_k + g_k) * masses[k]

    # Thrust required.
    F_x_thrust_k = F_x_k - F_x_aero_k
    F_y_thrust_k = F_y_k - F_y_aero_k

    # Thrust components.
    F_perpendicular_thrust_k = F_x_thrust_k * np.sin(gamma_r_interp[k]) - F_y_thrust_k * np.cos(gamma_r_interp[k])
    F_parallel_thrust_k = F_x_thrust_k * np.cos(gamma_r_interp[k]) + F_y_thrust_k * np.sin(gamma_r_interp[k])

    # Required angular acceleration.
    theta_dot_dot_k = 1/dt**2 * (gamma_r_interp[k+1] - 2*gamma_r_interp[k] + gamma_r_interp[k-1])    

    # Number of engines worth at full throttle without losses.
    n_e_t = math.sqrt(F_parallel_thrust_k**2 + F_perpendicular_thrust_k**2) / T_e_losses_k

    # Fixed line
    mdot_k = n_e_t / v_ex * T_e_nolosses

    # Propellant mass update.   
    m_p[k+1] = m_p[k] - mdot_k * dt # Different scale
    chi_k = (m_p_0 - m_p[k+1]) / m_p_0

    # Center of gravity and inertia.
    x_cog_k, I_z_k = cog_inertia_func(1 - chi_k)

    # Angular dynamics.
    M_z_k = theta_dot_dot_k * I_z_k

    # Center of pressure.
    d_cp_cg_k = cop_k - x_cog_k

    # Aero moment.
    M_z_aero_k = d_cp_cg_k * (F_y_aero_k * np.cos(gamma_r_interp[k]) - F_x_aero_k * np.sin(gamma_r_interp[k]))

    # Thrust moment.
    M_z_thrust_k = M_z_k - M_z_aero_k

    # Mass update.
    masses[k+1] = masses[k] - mdot_k * dt
    masses_sims[k+1] = masses_sims[k] - mdot_k * dt
    # Clipping removed for now.
    clip_bool = False
    if clip_bool:
        u0_k = np.clip(M_z_thrust_k / maximum_Mz_moment, -1, 1)
        u1_k = np.clip((F_parallel_thrust_k - maximum_F_parallel_thrust * minimum_F_parallel_thrust_factor) / (maximum_F_parallel_thrust * (1 - minimum_F_parallel_thrust_factor)), 0, 1) - 1
        u2_k = np.clip(F_perpendicular_thrust_k / maximum_F_perpendicular_thrust, -1, 1)
    else:
        u0_k = M_z_thrust_k / maximum_Mz_moment
        u1_k = (F_parallel_thrust_k - maximum_F_parallel_thrust * minimum_F_parallel_thrust_factor) / (maximum_F_parallel_thrust * (1 - minimum_F_parallel_thrust_factor))
        u2_k = F_perpendicular_thrust_k / maximum_F_perpendicular_thrust

    actions_k = np.array([u0_k, u1_k, u2_k])
    actions[k] = actions_k


# Check results
physics_step_lambda, initial_physics_state = compile_physics(dt)
states = np.zeros((len(times_r_interp), len(initial_physics_state)))
states[0] = initial_physics_state

x_actual = np.zeros(len(times_r_interp))
y_actual = np.zeros(len(times_r_interp))
vx_actual = np.zeros(len(times_r_interp))
vy_actual = np.zeros(len(times_r_interp))
u0_actual = np.zeros(len(times_r_interp))
u1_actual = np.zeros(len(times_r_interp))
u2_actual = np.zeros(len(times_r_interp))


for k in range(len(times_r_interp) - 1):
    states[k+1], _ = physics_step_lambda(states[k], actions[k])
    x_actual[k+1] = states[k+1][0]
    y_actual[k+1] = states[k+1][1]
    vx_actual[k+1] = states[k+1][2]
    vy_actual[k+1] = states[k+1][3]
    u0_actual[k+1] = actions[k][0]
    u1_actual[k+1] = actions[k][1]
    u2_actual[k+1] = actions[k][2]

# Create a figure and subplots with shared x-axis
fig, axs = plt.subplots(3, 3, figsize=(10, 5), sharex=True)

axs[0, 0].plot(times_r_interp, x_r_interp, label='Reference', linestyle='--', color='red')
axs[0, 0].plot(times_r_interp, x_actual, label='Actual', linestyle='-', color='blue')
axs[0, 0].plot(times_r_interp, x_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('x [m]')
axs[0, 0].grid()
axs[0, 0].legend()

axs[0, 1].plot(times_r_interp, y_r_interp, label='Reference', linestyle='--', color='red')
axs[0, 1].plot(times_r_interp, y_actual, label='Actual', linestyle='-', color='blue')
axs[0, 1].plot(times_r_interp, y_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('y [m]')
axs[0, 1].grid()
axs[0, 1].legend()

axs[0, 2].plot(times_r_interp, u0_actual, label='u0', linestyle='-', color='blue')
axs[0, 2].set_xlabel('Time [s]')
axs[0, 2].set_ylabel('u0')
axs[0, 2].grid()
axs[0, 2].legend()

axs[1, 0].plot(times_r_interp, vx_r_interp, label='Reference', linestyle='--', color='red')
axs[1, 0].plot(times_r_interp, vx_actual, label='Actual', linestyle='-', color='blue')
axs[1, 0].plot(times_r_interp, vx_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('vx [m/s]')
axs[1, 0].grid()
axs[1, 0].legend()

axs[1, 1].plot(times_r_interp, vy_r_interp, label='Reference', linestyle='--', color='red')
axs[1, 1].plot(times_r_interp, vy_actual, label='Actual', linestyle='-', color='blue')
axs[1, 1].plot(times_r_interp, vy_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel('vy [m/s]')
axs[1, 1].grid()
axs[1, 1].legend()

axs[1, 2].plot(times_r_interp, u1_actual, label='u1', linestyle='-', color='blue')
axs[1, 2].set_xlabel('Time [s]')
axs[1, 2].set_ylabel('u1')
axs[1, 2].grid()
axs[1, 2].legend()

axs[2, 0].plot(times_r_interp, masses_r_interp, label='Reference', linestyle='--', color='red')
axs[2, 0].plot(times_r_interp, masses, label='Actual', linestyle='-', color='blue')
axs[2, 0].plot(times_r_interp, masses_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[2, 0].set_xlabel('Time [s]')
axs[2, 0].set_ylabel('Mass [kg]')
axs[2, 0].grid()
axs[2, 0].legend()

axs[2, 1].plot(times_r_interp, np.rad2deg(gamma_r_interp), label='Reference', linestyle='--', color='red')
axs[2, 1].plot(times_r_interp, gamma_deg_sims, label='Simulation', linestyle=':', linewidth=5.5, color='green')
axs[2, 1].set_xlabel('Time [s]')
axs[2, 1].set_ylabel('gamma [deg]')
axs[2, 1].grid()
axs[2, 1].legend()

axs[2, 2].plot(times_r_interp, u2_actual, label='u2', linestyle='-', color='blue')
axs[2, 2].set_xlabel('Time [s]')
axs[2, 2].set_ylabel('u2')
axs[2, 2].grid()
axs[2, 2].legend()

plt.tight_layout()
plt.show()