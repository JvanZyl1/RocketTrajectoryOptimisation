import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import csv
import dill
from src.TrajectoryGeneration.Transformations import calculate_flight_path_angles

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD


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
dt = 0.01
times_r_interp = np.arange(times_r[0], times_r[-1], dt)
x_r_interp = interp.interp1d(times_r, x_r)(times_r_interp)
y_r_interp = interp.interp1d(times_r, y_r)(times_r_interp)
vx_r_interp = interp.interp1d(times_r, vx_r)(times_r_interp)
vy_r_interp = interp.interp1d(times_r, vy_r)(times_r_interp)
gamma_r_interp = np.deg2rad(calculate_flight_path_angles(vx_r_interp, vy_r_interp))

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
m_p_0 = float(sizing_results['Propellant mass stage 1 (ascent)'])*1000
cop_func = rocket_functions['cop_subrocket_0_lambda']
cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda']
maximum_Mz_moment = 0.75e9
maximum_F_parallel_thrust = 1.1e8
maximum_F_perpendicular_thrust = 1.75e7
minimum_F_parallel_thrust_factor = 0.7
m_p = np.zeros(len(times_r_interp))
m_p[0] = m_p_0

actions = np.zeros((len(times_r_interp) - 1, 3))

for k in range(len(times_r_interp) - 1):
    g_k = gravity_model_endo(y_r_interp[k])
    rho_k, p_k, a_k = endo_atmospheric_model(y_r_interp[k])

    vdot_x_k = (vx_r_interp[k+1] - vx_r_interp[k]) / dt
    vdot_y_k = (vy_r_interp[k+1] - vy_r_interp[k]) / dt

    F_x_k = vdot_x_k * masses[k]
    F_y_k = (vdot_y_k + g_k) * masses[k]

    speed_k = np.sqrt(vx_r_interp[k]**2 + vy_r_interp[k]**2)
    mach_k = speed_k / a_k

    C_D_k = CD_func(0, mach_k)
    C_L_k = CL_func(0, mach_k)

    L_k = 0.5 * rho_k * speed_k**2 * C_L_k * frontal_area
    D_k = 0.5 * rho_k * speed_k**2 * C_D_k * frontal_area

    F_x_aero_k = - D_k * np.cos(gamma_r_interp[k]) - L_k * np.cos(math.pi - gamma_r_interp[k])
    F_y_aero_k = - D_k * np.sin(gamma_r_interp[k]) + L_k * np.sin(math.pi - gamma_r_interp[k])

    F_x_thrust_k = F_x_k - F_x_aero_k
    F_y_thrust_k = F_y_k - F_y_aero_k

    F_perpendicular_thrust_k = F_x_thrust_k * np.sin(gamma_r_interp[k]) - F_y_thrust_k * np.cos(gamma_r_interp[k])
    F_parallel_thrust_k = F_x_thrust_k * np.cos(gamma_r_interp[k]) + F_y_thrust_k * np.sin(gamma_r_interp[k])

    theta_dot_dot_k = 1/dt**2 * (gamma_r_interp[k+1] - 2*gamma_r_interp[k] + gamma_r_interp[k-1])

    T_e_losses_k = T_e_nolosses + (p_e - p_k) * A_e

    n_e_t = math.sqrt(F_parallel_thrust_k**2 + F_perpendicular_thrust_k**2) / T_e_losses_k

    mdot_k = n_e_t * T_e_losses_k / v_ex

    m_p[k+1] = m_p[k] - mdot_k * dt # Different scale
    chi_k = (m_p_0 - m_p[k+1]) / m_p_0

    x_cog_k, I_z_k = cog_inertia_func(1 - chi_k)

    M_z_k = theta_dot_dot_k * I_z_k

    cop_k = cop_func(0, mach_k)

    d_cp_cg_k = cop_k - x_cog_k

    M_z_aero_k = d_cp_cg_k * (F_y_aero_k * np.cos(gamma_r_interp[k]) - F_x_aero_k * np.sin(gamma_r_interp[k]))

    M_z_thrust_k = M_z_k - M_z_aero_k

    masses[k+1] = masses[k] - mdot_k * dt

    u0_k = np.clip(M_z_thrust_k / maximum_Mz_moment, -1, 1)
    u1_k = np.clip((F_parallel_thrust_k - maximum_F_parallel_thrust * minimum_F_parallel_thrust_factor) / (maximum_F_parallel_thrust * (1 - minimum_F_parallel_thrust_factor)), 0, 1) - 1
    u2_k = np.clip(F_perpendicular_thrust_k / maximum_F_perpendicular_thrust, -1, 1)

    actions_k = np.array([u0_k, u1_k, u2_k])
    actions[k] = actions_k