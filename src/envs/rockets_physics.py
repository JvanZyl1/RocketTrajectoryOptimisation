import csv
import dill
import math
import numpy as np

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD

def rocket_physics_fcn(state,
                      actions,
                      # Lambda wrapped
                      dt,
                      initial_propellant_mass,
                      cog_inertia_func,
                      d_thrust_cg_func,
                      cop_func,
                      frontal_area,
                      v_exhaust,
                      nozzle_exit_area,
                      nozzle_exit_pressure,
                      thrust_per_engine,
                      number_of_engines_gimballed,
                      number_of_engines_non_gimballed,
                      CL_func,
                      CD_func,
                      maximum_Mz_moment = 0.75e9,
                      maximum_F_parallel_thrust = 1.1e8,
                      maximum_F_perpendicular_thrust = 1.75e7,
                      minimum_F_parallel_thrust_factor = 0.7):
    # van-Kampen style action augmentation
    u0, u1, u2 = actions
    # HARDCODED VALUES atm with slack for extra control authority later on
    # u0 relates to the moment around the z-axis
    M_z_thrust= np.clip(u0, -1, 1) * maximum_Mz_moment
    # u1 relates to force parallel to the rocket axis
    F_parallel_thrust = np.clip(u1 + 1, 0, 1) * maximum_F_parallel_thrust * (1 - minimum_F_parallel_thrust_factor) + minimum_F_parallel_thrust_factor * maximum_F_parallel_thrust
    # u2 relates to force perpendicular to the rocket axis
    F_perpendicular_thrust = np.clip(u2, -1, 1) * maximum_F_perpendicular_thrust

    # Unpack state
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Atmopshere values
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
    speed = math.sqrt(vx**2 + vy**2)
    mach_number = speed / speed_of_sound

    # Gravity
    g = gravity_model_endo(y)

    # Calculate dynamic pressure
    dynamic_pressure = 0.5 * density * speed**2

    # Determine later whether to do with Mach number of angle of attack
    C_L = CL_func(alpha, mach_number)
    C_D = CD_func(alpha, mach_number)
    CoP = cop_func(math.degrees(alpha), mach_number)

    # Lift and drag
    drag = 0.5 * density * speed**2 * C_D * frontal_area
    lift = 0.5 * density * speed**2 * C_L * frontal_area
    aero_x = -drag * math.cos(gamma) - lift * math.cos(math.pi - gamma)
    aero_y = -drag * math.sin(gamma) + lift * math.sin(math.pi - gamma)

    # thrusts
    thrust_engine_with_losses_full_throttle = (thrust_per_engine + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)

    total_thrust = np.sqrt(F_parallel_thrust**2 + F_perpendicular_thrust**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    if number_of_engines_thrust_total > (number_of_engines_gimballed + number_of_engines_non_gimballed):
        raise Warning("Thrust too high.")
    
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

    acceleration_dict = {
        'acceleration_x_component_thrust': thrust_x/mass,
        'acceleration_y_component_thrust': thrust_y/mass,
        'acceleration_x_component_drag': -drag * math.cos(gamma)/mass,
        'acceleration_y_component_drag': -drag/mass * math.sin(gamma)/mass,
        'acceleration_x_component_lift': - lift * math.cos(math.pi - gamma)/mass,
        'acceleration_y_component_lift': lift * math.sin(math.pi - gamma)/mass,
        'acceleration_x_component_gravity': 0,
        'acceleration_y_component_gravity': -g,
        'acceleration_x_component': vx_dot,
        'acceleration_y_component': vy_dot
    }

    # Tank fill level
    mass_propellant -= mass_flow * dt
    fuel_percentage_consumed = (initial_propellant_mass - mass_propellant) / initial_propellant_mass
    
    # x_cog and inertia
    x_cog, inertia = cog_inertia_func(1-fuel_percentage_consumed)

    # center of pressure
    d_cp_cg = CoP - x_cog

    # thrust displacement from cog : logging only
    d_thrust_cg = d_thrust_cg_func(x_cog)

    # Angular dynamics
    aero_moments_z = (-aero_x * math.sin(theta) + aero_y * math.cos(theta)) * d_cp_cg
    moments_z = M_z_thrust + aero_moments_z 
    theta_dot_dot = moments_z / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = math.atan2(vy, vx)

    if theta > 2 * math.pi:
        theta -= 2 * math.pi

    alpha = theta - gamma

    mass -= mass_flow * dt

    time += dt

    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]

    moments_dict = {
        'thrust_moments_z': M_z_thrust,
        'aero_moements_z': aero_moments_z,
        'moments_z': moments_z,
        'theta_dot_dot': theta_dot_dot
    }
    
    info = {
        'inertia': inertia,
        'acceleration_dict': acceleration_dict,
        'mach_number': mach_number,
        'CL': C_L,
        'CD': C_D,
        'drag': drag,
        'lift': lift,
        'moment_dict': moments_dict,
        'd_cp_cg': d_cp_cg,
        'd_thrust_cg': d_thrust_cg,
        'x_cog': x_cog,
        'dynamic_pressure': dynamic_pressure,
        'number_of_engines_thrust_total': number_of_engines_thrust_total,
        'mass_flow': mass_flow,
        'fuel_percentage_consumed': fuel_percentage_consumed,
        'F_parallel_thrust': F_parallel_thrust,
        'F_perpendicular_thrust': F_perpendicular_thrust,
        'thrust_x': thrust_x,
        'thrust_y': thrust_y,
        'thrust_engine_with_losses_full_throttle': thrust_engine_with_losses_full_throttle,        
    }
    
    return state, info


def compile_physics(dt,
                    kl_sub = 2.0,
                    kl_sup = 1.0,
                    cd0_subsonic=0.05,
                    kd_subsonic=0.5,
                    cd0_supersonic=0.10,
                    kd_supersonic=1.0):
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
    number_of_engines_gimballed_stage_1 = int(sizing_results['Number of engines gimballed stage 1'])
    number_of_engines_stage_1 = int(sizing_results['Number of engines stage 1'])
    number_of_engines_non_gimballed_stage_1 = number_of_engines_stage_1 - number_of_engines_gimballed_stage_1
    physics_step_lambda = lambda state, actions: \
            rocket_physics_fcn(state = state,
                               actions = actions,
                               dt = dt,
                               initial_propellant_mass = float(sizing_results['Propellant mass stage 1 (ascent)'])*1000,
                               cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                               d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_0_lambda'],
                               cop_func = rocket_functions['cop_subrocket_0_lambda'],
                               frontal_area = float(sizing_results['Rocket frontal area']),
                               v_exhaust = float(sizing_results['Exhaust velocity stage 1']),
                               nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                               nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                               thrust_per_engine = float(sizing_results['Thrust engine stage 1']),
                               number_of_engines_gimballed = number_of_engines_gimballed_stage_1,
                               number_of_engines_non_gimballed = number_of_engines_non_gimballed_stage_1,
                               CL_func = CL_func,
                               CD_func = CD_func,
                               maximum_Mz_moment = 0.75e9, # TODO - make automatic
                               maximum_F_parallel_thrust = 1.1e8,
                               maximum_F_perpendicular_thrust = 1.75e7,
                               minimum_F_parallel_thrust_factor = 0.7)
    # Initial physics state : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
    initial_physics_state = np.array([0,                                                        # x [m]
                                      0,                                                        # y [m]
                                      0,                                                        # vx [m/s]
                                      0,                                                        # vy [m/s]
                                      np.pi/2,                                                  # theta [rad]
                                      0,                                                        # theta_dot [rad/s]
                                      0,                                                        # gamma [rad]
                                      0,                                                        # alpha [rad]
                                      float(sizing_results['Initial mass (subrocket 0)'])*1000,             # mass [kg]
                                      float(sizing_results['Propellant mass stage 1 (ascent)'])*1000,       # mass_propellant [kg]
                                      0])                                                       # time [s]

    return physics_step_lambda, initial_physics_state