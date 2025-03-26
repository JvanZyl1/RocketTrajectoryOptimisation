import csv
import dill
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.NetworkFitting.utils.atmosphere import endo_atmospheric_model, gravity_model_endo
from src.NetworkFitting.utils.Aero_coeffs import rocket_CL, rocket_CD

def get_dt():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo.csv')
    # Extract time and state columns
    time = data['t[s]']
    dt_array = np.diff(time)
    dt = np.mean(dt_array)
    return dt


def reference_trajectory_lambda_func_y():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')
    
    # Extract y and state columns
    y_values = data['y[m]']
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values
    
    # Create an interpolation function for each state variable based on y
    interpolators = [interp1d(y_values, states[:, i], kind='linear', fill_value="extrapolate") for i in range(states.shape[1])]
    
    # Return a function that takes in a y value and returns the state
    def interpolate_state(y):
        result = np.array([interpolator(y) for interpolator in interpolators])
        if np.any(np.isnan(result)):
            print(f"NaN detected in interpolation result at y = {y}: {result}")
        return result
    
    terminal_state = states[-1]
    
    return interpolate_state, terminal_state

def determine_force_and_moment_required(state_r_0,
                                        state_r_1,
                                        theta_dot_minus_1,
                                        mass_propellant,
                                        dt,
                                        frontal_area,
                                        cog_inertia_func,
                                        initial_propellant_mass,
                                        thrust_per_engine,
                                        v_exhaust,
                                        nozzle_exit_area,
                                        nozzle_exit_pressure,
                                        cop_func,
                                        CL_func,
                                        CD_func):
    x_0, y_0, vx_0, vy_0, mass_0 = state_r_0
    x_1, y_1, vx_1, vy_1, mass_1 = state_r_1

    # Lift and drag
    density_0, atmospheric_pressure_0, speed_of_sound_0 = endo_atmospheric_model(y_0)
    speed_0 = math.sqrt(vx_0**2 + vy_0**2)
    mach_number_0 = speed_0 / speed_of_sound_0
    C_D = CD_func(0, mach_number_0)
    C_L = CL_func(0, mach_number_0)
    gamma_0 = math.atan2(vy_0, vx_0)
    drag = 0.5 * density_0 * speed_0**2 * C_D * frontal_area
    lift = 0.5 * density_0 * speed_0**2 * C_L * frontal_area
    aero_x = -drag * math.cos(gamma_0) - lift * math.cos(math.pi - gamma_0)
    aero_y = -drag * math.sin(gamma_0) + lift * math.sin(math.pi - gamma_0)

    # Gravity
    g_0 = gravity_model_endo(y_0)

    # Velocities
    vx_dot_0 = (vx_1 - vx_0) / dt
    vy_dot_0 = (vy_1 - vy_0) / dt

    # Forces
    forces_x = mass_0 * vx_dot_0
    forces_y = mass_0 * vy_dot_0 - g_0    

    # Thrusts
    thrust_x = forces_x - aero_x
    thrust_y = forces_y - aero_y

    # Forces
    theta_0 = gamma_0  # Alpha = 0
    F_perpendicular_thrust = math.cos(theta_0)/(math.cos(theta_0)**2 + math.sin(theta_0)**2) * (
        thrust_x * math.sin(theta_0)/math.cos(theta_0) - thrust_y
    )
    F_parallel_thrust = 1/math.cos(theta_0) * (
        thrust_x - F_perpendicular_thrust * math.sin(theta_0)
    )

    # u1, u2
    u2 = F_perpendicular_thrust / 1.75e7
    F_parallel_thrust_max = 1.1e8
    u1 = (F_parallel_thrust - 0.7 * F_parallel_thrust_max) / (0.3 * F_parallel_thrust_max) - 1


    # Mass flow
    thrust_engine_with_losses_full_throttle = (thrust_per_engine + (nozzle_exit_pressure - atmospheric_pressure_0) * nozzle_exit_area)
    total_thrust = np.sqrt(F_parallel_thrust**2 + F_perpendicular_thrust**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine / v_exhaust) * number_of_engines_thrust_total
    mass_propellant -= mass_flow * dt

    # Inertia
    fuel_percentage_consumed = (initial_propellant_mass - mass_propellant) / initial_propellant_mass
    x_cog_0, inertia_0 = cog_inertia_func(1-fuel_percentage_consumed)


    # Aerodynamic moments
    CoP_0 = cop_func(0, mach_number_0)
    d_cp_cg = CoP_0 - x_cog_0
    aero_moments_z = (-aero_x * math.sin(theta_0) + aero_y * math.cos(theta_0)) * d_cp_cg

    # Angular dynamics
    gamma_1 = math.atan2(vy_1, vx_1)
    theta_dot_0 = (gamma_1 - gamma_0) / dt   # theta = gamma if alpha is 0
    theta_dot_dot_0 = (theta_dot_0 - theta_dot_minus_1) / dt
    moments_z = theta_dot_dot_0 * inertia_0
    M_z_thrust = moments_z - aero_moments_z

    # u0
    u0 = M_z_thrust / 0.75e9

    return u0, u1, u2, theta_dot_0

def physics_validation_step(state,
                      actions,
                      # Lambda wrapped
                      dt,
                      initial_propellant_mass,
                      cog_inertia_func,
                      cop_func,
                      frontal_area,
                      v_exhaust,
                      nozzle_exit_area,
                      nozzle_exit_pressure,
                      thrust_per_engine,
                      CL_func,
                      CD_func):
    # van-Kampen style action augmentation
    u0, u1, u2 = actions
    M_z_thrust= u0 * 0.75e9
    F_parallel_thrust_max = 1.1e8
    F_parallel_thrust = (u1 + 1) * F_parallel_thrust_max * 0.3 + 0.7 * F_parallel_thrust_max
    F_perpendicular_thrust = u2 * 1.75e7

    # Unpack state
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    print(f'Altitude: {y}')

    # Atmospheric values
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
    speed = math.sqrt(vx**2 + vy**2)
    mach_number = speed / speed_of_sound
    g = gravity_model_endo(y)

    # Aero coefficients
    C_L = CL_func(alpha, mach_number)
    C_D = CD_func(alpha, mach_number)
    CoP = cop_func(math.degrees(alpha), mach_number)

    # Lift and drag
    drag = 0.5 * density * speed**2 * C_D * frontal_area
    lift = 0.5 * density * speed**2 * C_L * frontal_area
    aero_x = -drag * math.cos(gamma) - lift * math.cos(math.pi - gamma)
    aero_y = -drag * math.sin(gamma) + lift * math.sin(math.pi - gamma)

    # Thrusts
    thrust_engine_with_losses_full_throttle = (thrust_per_engine + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    total_thrust = np.sqrt(F_parallel_thrust**2 + F_perpendicular_thrust**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine / v_exhaust) * number_of_engines_thrust_total
    thrust_x = (F_parallel_thrust) * math.cos(theta) + F_perpendicular_thrust * math.sin(theta)
    thrust_y = (F_parallel_thrust) * math.sin(theta) - F_perpendicular_thrust * math.cos(theta)

    # Forces
    forces_x = aero_x + thrust_x
    forces_y = aero_y + thrust_y

    # Kinematics
    vx_dot = forces_x/mass
    vy_dot = forces_y/mass - g
    vx += vx_dot * dt
    vy += vy_dot * dt
    x += vx * dt
    y += vy * dt

    # Tank fill level
    mass_propellant -= mass_flow * dt
    fuel_percentage_consumed = (initial_propellant_mass - mass_propellant) / initial_propellant_mass
    
    # x_cog and inertia
    x_cog, inertia = cog_inertia_func(1-fuel_percentage_consumed)

    # center of pressure
    d_cp_cg = CoP - x_cog

    # Angular dynamics
    aero_moments_z = (-aero_x * math.sin(theta) + aero_y * math.cos(theta)) * d_cp_cg
    moments_z = M_z_thrust + aero_moments_z 
    theta_dot_dot = moments_z / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = math.atan2(vy, vx)
    alpha = theta - gamma

    # Mass
    mass -= mass_flow * dt

    # Time
    time += dt

    # State
    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    
    return state


def validate_actions(u0_list,
                     u1_list,
                     u2_list,
                     time_list,
                     reference_states,
                     physics_validation_step_lambda,
                     mass_propellant0):
    
    x = []
    y = []
    vx = []
    vy = []
    alpha = []
    mass = []
    times = []


    x0, y0, vx0, vy0, mass0 = reference_states[0]
    state = [x0, y0, vx0, vy0, math.radians(90), 0, 0, 0, mass0, mass_propellant0, 0]
    for i in range(len(u0_list) - 1):
        actions = (u0_list[i], u1_list[i], u2_list[i])
        dt = time_list[i+1] - time_list[i]
        state = physics_validation_step_lambda(state, actions, dt)
        x.append(state[0])
        y.append(state[1])
        vx.append(state[2])
        vy.append(state[3])
        alpha.append(state[5])
        mass.append(state[8])
        times.append(state[-1])
    xr = []
    yr = []
    vxr = []
    vyr = []
    massr = []
    alphar = []
    for i in range(len(reference_states) - 1):
        reference_state = reference_states[i]
        xr.append(reference_state[0])
        yr.append(reference_state[1])
        vxr.append(reference_state[2])
        vyr.append(reference_state[3])
        massr.append(reference_state[4])

        alphar.append(0)
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 2, 1)
    plt.plot(times, x, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, xr, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('x [m]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(times, y, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, yr, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(times, vx, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, vxr, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('vx [m/s]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(times, vy, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, vyr, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('vy [m/s]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(times, alpha, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, alphar, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('alpha [deg]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(times, mass, linewidth=2.0, linestyle='-', label='Actual')
    plt.plot(times, massr, linewidth=2.0, linestyle='--', label='Reference')
    plt.xlabel('Time [s]')
    plt.ylabel('mass [kg]')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def lambda_wrap_determine_force_and_moment_required():
    kl_sub = 2.0
    kl_sup = 1.0
    cd0_subsonic=0.05
    kd_subsonic=0.5
    cd0_supersonic=0.10
    kd_supersonic=1.0
    CL_func = lambda alpha, M: rocket_CL(alpha, M, kl_sub, kl_sup)
    CD_func = lambda alpha, M: rocket_CD(alpha, M, cd0_subsonic, kd_subsonic, cd0_supersonic, kd_supersonic)

    # Read sizing results
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]

    with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
        rocket_functions = dill.load(f)


    determine_force_and_moment_required_lambda = lambda state_r_0, state_r_1, theta_dot_minus_1, mass_propellant, dt: \
        determine_force_and_moment_required(state_r_0,
                                            state_r_1,
                                            theta_dot_minus_1,
                                            mass_propellant,
                                            dt,
                                            float(sizing_results['Rocket frontal area']),
                                            rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                                            float(sizing_results['Propellant mass stage 1 (ascent)'])*1000,
                                            float(sizing_results['Thrust engine stage 1']),
                                            float(sizing_results['Exhaust velocity stage 1']),
                                            float(sizing_results['Nozzle exit area']),
                                            float(sizing_results['Nozzle exit pressure stage 1']),
                                            rocket_functions['cop_subrocket_0_lambda'],
                                            CL_func,
                                            CD_func)
    
    physics_validation_step_lambda = lambda state, actions, dt: \
        physics_validation_step(state,
                                actions,
                                dt,
                                initial_propellant_mass = float(sizing_results['Propellant mass stage 1 (ascent)'])*1000,
                                cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                                cop_func = rocket_functions['cop_subrocket_0_lambda'],
                                frontal_area = float(sizing_results['Rocket frontal area']),
                                v_exhaust = float(sizing_results['Exhaust velocity stage 1']),
                                nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                                thrust_per_engine = float(sizing_results['Thrust engine stage 1']),
                                CL_func = CL_func,
                                CD_func = CD_func)
    
    validate_actions_lambda = lambda u0_list, u1_list, u2_list, time_list, reference_states: \
        validate_actions(u0_list, 
                         u1_list, 
                         u2_list, 
                         time_list, 
                         reference_states, 
                         physics_validation_step_lambda,
                         float(sizing_results['Propellant mass stage 1 (ascent)'])*1000)

    return determine_force_and_moment_required_lambda, float(sizing_results['Propellant mass stage 1 (ascent)'])*1000, \
        validate_actions_lambda
def find_required_actions():
    data = pd.read_csv('data/reference_trajectory/reference_trajectory_endo_clean.csv')

    action_determination_lambda_func, mass_propellant, validate_actions_lambda = lambda_wrap_determine_force_and_moment_required()

    theta_dot_minus_1 = 0
    u0_list = []
    u1_list = []
    u2_list = []
    time_list = []
    reference_states = []

    for i in range(len(data) - 1):
        state_r_0 = data.iloc[i][['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values
        state_r_1 = data.iloc[i+1][['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'mass[kg]']].values

        state_r_0_list = state_r_0.tolist()
        state_r_1_list = state_r_1.tolist()
        reference_states.append(state_r_0_list)
        dt = data.iloc[i+1]['t[s]'] - data.iloc[i]['t[s]']
        u0, u1, u2, theta_dot_minus_1 = action_determination_lambda_func(state_r_0_list, state_r_1_list, theta_dot_minus_1, mass_propellant, dt)
        u0_list.append(u0)
        u1_list.append(u1)
        u2_list.append(u2)
        time_list.append(data.iloc[i]['t[s]'])

    u0_non_smooth = u0_list.copy()
    u1_non_smooth = u1_list.copy()
    u2_non_smooth = u2_list.copy()

    # Some manual alteration, every entry between 5 and 9 seconds is set to the interpolated value between 5 and 9 seconds.
    idx_9s = np.argmin(np.abs(np.array(time_list) - 9))
    idx_5s = np.argmin(np.abs(np.array(time_list) - 5))
    u0_list[idx_5s:idx_9s] = np.interp(time_list[idx_5s:idx_9s], [5, 9], [u0_list[idx_5s], u0_list[idx_9s]])
    u1_list[idx_5s:idx_9s] = np.interp(time_list[idx_5s:idx_9s], [5, 9], [u1_list[idx_5s], u1_list[idx_9s]])
    u2_list[idx_5s:idx_9s] = np.interp(time_list[idx_5s:idx_9s], [5, 9], [u2_list[idx_5s], u2_list[idx_9s]])


    # First entries
    for i in range(5):
        u0_list[i] = u0_list[5]
        u1_list[i] = u1_list[5]
        u2_list[i] = u2_list[5]

    # 39 to 39.5
    idx_39s = np.argmin(np.abs(np.array(time_list) - 39))
    idx_395s = np.argmin(np.abs(np.array(time_list) - 39.5))
    u0_list[idx_39s:idx_395s] = np.interp(time_list[idx_39s:idx_395s], [39, 39.5], [u0_list[idx_39s], u0_list[idx_395s]])
    u1_list[idx_39s:idx_395s] = np.interp(time_list[idx_39s:idx_395s], [39, 39.5], [u1_list[idx_39s], u1_list[idx_395s]])
    u2_list[idx_39s:idx_395s] = np.interp(time_list[idx_39s:idx_395s], [39, 39.5], [u2_list[idx_39s], u2_list[idx_395s]])

    # 83.7 to 84.2
    idx_837s = np.argmin(np.abs(np.array(time_list) - 83.7))
    idx_842s = np.argmin(np.abs(np.array(time_list) - 84.2))
    u0_list[idx_837s:idx_842s] = np.interp(time_list[idx_837s:idx_842s], [83.7, 84.2], [u0_list[idx_837s], u0_list[idx_842s]])
    u1_list[idx_837s:idx_842s] = np.interp(time_list[idx_837s:idx_842s], [83.7, 84.2], [u1_list[idx_837s], u1_list[idx_842s]])
    u2_list[idx_837s:idx_842s] = np.interp(time_list[idx_837s:idx_842s], [83.7, 84.2], [u2_list[idx_837s], u2_list[idx_842s]])


    # 105 to 106
    idx_105s = np.argmin(np.abs(np.array(time_list) - 105))
    idx_106s = np.argmin(np.abs(np.array(time_list) - 106))
    u0_list[idx_105s:idx_106s] = np.interp(time_list[idx_105s:idx_106s], [105, 106], [u0_list[idx_105s], u0_list[idx_106s]])
    u1_list[idx_105s:idx_106s] = np.interp(time_list[idx_105s:idx_106s], [105, 106], [u1_list[idx_105s], u1_list[idx_106s]])
    u2_list[idx_105s:idx_106s] = np.interp(time_list[idx_105s:idx_106s], [105, 106], [u2_list[idx_105s], u2_list[idx_106s]])

    # Last entry interp
    u0_list[-1] = u0_list[-2]
    u1_list[-1] = u1_list[-2]
    u2_list[-1] = u2_list[-2]
        
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(time_list, u0_list, linewidth=2.0, linestyle='-', label='u0')
    plt.xlabel('Time [s]')
    plt.ylabel('u0 : Moment z')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_list, u1_list, linewidth=2.0, linestyle='-', label='u1')
    plt.xlabel('Time [s]')
    plt.ylabel('u1 : Thrust parallel')
    plt.ylim(-3.5, 0)
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_list, u2_list, linewidth=2.0, linestyle='-', label='u2')
    plt.xlabel('Time [s]')
    plt.ylabel('u2 : Thrust perpendicular')
    plt.ylim(-1, 1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.close()


    validate_actions_lambda(u0_non_smooth, u1_non_smooth, u2_non_smooth, time_list, reference_states)

    

if __name__ == "__main__":
    find_required_actions()