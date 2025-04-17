import csv
import dill
import math
import numpy as np
from control import tf

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD

def force_moment_decomposer_ascent(actions,
                                   atmospheric_pressure : float,
                                   d_thrust_cg : float,
                                   thrust_per_engine_no_losses : float,
                                   nozzle_exit_pressure : float,
                                   nozzle_exit_area : float,
                                   number_of_engines_gimballed : int,
                                   number_of_engines_non_gimballed : int,
                                   v_exhaust : float,
                                   nominal_throttle : float = 0.5,
                                   max_gimbal_angle_rad : float = math.radians(1)):
    # Actions : u0, u1
    # u0 is gimbal angle norm from -1 to 1
    # u1 is non nominal throttle from -1 to 1
    u0, u1 = actions
    gimbal_angle_rad = u0 * max_gimbal_angle_rad

    non_nominal_throttle = (u1 + 1) / 2
    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle
    thrust_non_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_non_gimballed * throttle

    thrust_parallel = thrust_non_gimballed + thrust_gimballed * math.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    if number_of_engines_thrust_total > (number_of_engines_gimballed + number_of_engines_non_gimballed)  + 0.01:
        raise Warning("Thrust too high.")
    
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = math.degrees(gimbal_angle_rad)
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle

def first_order_low_pass_step(x, u, tau, dt):
    dx = (-x + u) / tau
    y = x + dt * dx
    return y


def force_moment_decomposer_flip_over(action,
                                      atmospheric_pressure,
                                      d_thrust_cg,
                                      gimbal_angle_deg_prev,
                                      max_gimbal_angle_deg, # 45
                                      thrust_per_engine_no_losses,
                                      nozzle_exit_pressure,
                                      nozzle_exit_area,
                                      number_of_engines_flip_over, # gimballed
                                      v_exhaust):
    gimbal_angle_command_deg = action * max_gimbal_angle_deg
    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = gimbal_angle_command_deg,
                                                 tau = 1.0,
                                                 dt = 0.1)
    gimbal_angle_rad = math.radians(gimbal_angle_deg)
    
    # No pressure losses but include for later graphs continuity.
    throttle = 1
    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_flip_over * throttle

    thrust_parallel = thrust_gimballed * math.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    if number_of_engines_thrust_total > (number_of_engines_flip_over) + 0.01: # numerical errors
        raise Warning("Thrust too high. Number of engines thrust total: " + str(number_of_engines_thrust_total) + " Number of engines flip over: " + str(number_of_engines_flip_over))
    
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = math.degrees(gimbal_angle_rad)
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg

def rocket_physics_fcn(state : np.array,
                      actions : np.array,
                      # Lambda wrapped
                      flight_phase : str,
                      control_function : callable,
                      dt : float,
                      initial_propellant_mass_stage : float,
                      cog_inertia_func : callable,
                      d_thrust_cg_func : callable,
                      cop_func : callable,
                      frontal_area : float,
                      CL_func : callable,
                      CD_func : callable,
                      gimbal_angle_deg_prev : float = None):
    # Unpack state
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Inertia and thrust displacement from cog
    fuel_percentage_consumed = (initial_propellant_mass_stage - mass_propellant) / initial_propellant_mass_stage
    if fuel_percentage_consumed == 0.0:
        fuel_percentage_consumed = 1e-6
    x_cog, inertia = cog_inertia_func(1-fuel_percentage_consumed)
    d_thrust_cg = d_thrust_cg_func(x_cog)

    # Atmopshere values
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
    speed = math.sqrt(vx**2 + vy**2)
    mach_number = speed / speed_of_sound

    if flight_phase in ['subsonic', 'supersonic']:
        thrust_parallel, thrust_perpendicular, moments_z_control, mass_flow, \
             gimbal_angle_deg, throttle = control_function(actions, atmospheric_pressure, d_thrust_cg)
        action_info = {
            'gimbal_angle_deg': gimbal_angle_deg,
            'throttle': throttle
        }
    elif flight_phase == 'flip_over':
        assert gimbal_angle_deg_prev is not None, "Gimbal angle degree previous is required for flip over"
        thrust_parallel, thrust_perpendicular, moments_z_control, mass_flow, gimbal_angle_deg = control_function(actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev)
        action_info = {
            'gimbal_angle_deg': gimbal_angle_deg
        }
    
    thrust_x = (thrust_parallel) * math.cos(theta) + thrust_perpendicular * math.sin(theta)
    thrust_y = (thrust_parallel) * math.sin(theta) - thrust_perpendicular * math.cos(theta)

    # Gravity
    g = gravity_model_endo(y)

    # Calculate dynamic pressure
    dynamic_pressure = 0.5 * density * speed**2

    # Determine later whether to do with Mach number of angle of attack
    C_L = CL_func(alpha, mach_number)
    C_D = CD_func(alpha, mach_number)
    CoP = cop_func(math.degrees(alpha), mach_number)
    d_cp_cg = CoP - x_cog

    # Lift and drag
    drag = 0.5 * density * speed**2 * C_D * frontal_area
    lift = 0.5 * density * speed**2 * C_L * frontal_area
    aero_x = -drag * math.cos(gamma) - lift * math.cos(math.pi - gamma)
    aero_y = -drag * math.sin(gamma) + lift * math.sin(math.pi - gamma)

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

    # Angular dynamics
    aero_moments_z = (-aero_x * math.sin(theta) + aero_y * math.cos(theta)) * d_cp_cg
    moments_z = moments_z_control + aero_moments_z 
    theta_dot_dot = moments_z / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = math.atan2(vy, vx)

    if theta > 2 * math.pi:
        theta -= 2 * math.pi

    alpha = theta - gamma

    # Mass Update
    mass_propellant -= mass_flow * dt
    mass -= mass_flow * dt

    time += dt

    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]


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

    moments_dict = {
        'thrust_moments_z': moments_z_control,
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
        'mass_flow': mass_flow,
        'fuel_percentage_consumed': fuel_percentage_consumed,
        'F_parallel_thrust': thrust_parallel,
        'F_perpendicular_thrust': thrust_perpendicular,
        'thrust_x': thrust_x,
        'thrust_y': thrust_y,
        'atmospheric_pressure': atmospheric_pressure,
        'air_density': density,
        'speed_of_sound': speed_of_sound,
        'action_info': action_info
    }

    return state, info


def compile_physics(dt,
                    flight_phase : str,
                    kl_sub = 2.0,
                    kl_sup = 1.0,
                    cd0_subsonic=0.05,
                    kd_subsonic=0.5,
                    cd0_supersonic=0.10,
                    kd_supersonic=1.0):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over']
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

    if flight_phase in ['subsonic', 'supersonic']:
        force_composer_lambda = lambda actions, atmospheric_pressure, d_thrust_cg : \
            force_moment_decomposer_ascent(actions, atmospheric_pressure, d_thrust_cg,
                                           thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
                                           nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                                           nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                           number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1']),
                                           number_of_engines_non_gimballed = int(sizing_results['Number of engines stage 1']) \
                                            - int(sizing_results['Number of engines gimballed stage 1']),
                                           v_exhaust = float(sizing_results['Exhaust velocity stage 1']),
                                           nominal_throttle = 0.5,
                                           max_gimbal_angle_rad = math.radians(1))

        physics_step_lambda = lambda state, actions: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = (float(sizing_results['Propellant mass stage 1 (ascent)']) \
                                                               + float(sizing_results['Propellant mass stage 1 (descent)']))*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_0_lambda'],
                                   cop_func = rocket_functions['cop_subrocket_0_lambda'],
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func)
    elif flight_phase == 'flip_over':
        force_composer_lambda = lambda actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev : \
            force_moment_decomposer_flip_over(actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev,
                                              max_gimbal_angle_deg=45,
                                              thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
                                              nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                                              nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                              number_of_engines_flip_over = 6,
                                              v_exhaust = float(sizing_results['Exhaust velocity stage 1']))
        physics_step_lambda = lambda state, actions, gimbal_angle_deg_prev: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = (float(sizing_results['Propellant mass stage 1 (ascent)']) \
                                                               + float(sizing_results['Propellant mass stage 1 (descent)']))*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda'],
                                   cop_func = rocket_functions['cop_subrocket_2_lambda'],
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   gimbal_angle_deg_prev = gimbal_angle_deg_prev)
    return physics_step_lambda

