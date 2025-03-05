import math
import numpy as np
import csv
import dill

from src.envs.utils.atmosphere import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.Aero_coeffs import rocket_CL, rocket_CD

# Vertical rising and gravity turn
def rocket_model_physics_step_endo(state,
                      actions,
                      propellant_mass,
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
                      CD_func):
    
    # Clip actions at the physics level
    actions = np.clip(actions, -1, 1)
    
    # x is through top of rocket, y is through side of rocket, z is to bottom
    # Unpack actions
    # actions is now guaranteed to be between (-1,1)
    u = actions
    ratio_force_gimballed_x = u * 0.2
    ratio_force_gimballed_y = 1 - abs(ratio_force_gimballed_x)

    # Unpack state
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass = state

    g_thrust_without_losses = thrust_per_engine * (number_of_engines_gimballed + number_of_engines_non_gimballed) / mass * 1/9.81

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
    mass_flow = thrust_per_engine * (number_of_engines_gimballed + number_of_engines_non_gimballed) / v_exhaust

    thrust_engine_with_losses = (thrust_per_engine + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_non_gimballed = thrust_engine_with_losses * number_of_engines_non_gimballed
    thrust_gimballed = thrust_engine_with_losses * number_of_engines_gimballed
    thrust_x = thrust_gimballed * ratio_force_gimballed_x + thrust_non_gimballed * math.cos(theta)
    thrust_y = thrust_gimballed * ratio_force_gimballed_y + thrust_non_gimballed * math.sin(theta)

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
    propellant_mass -= mass_flow * dt
    fuel_percentage_consumed = (initial_propellant_mass - propellant_mass) / initial_propellant_mass
    
    # x_cog and inertia
    x_cog, inertia = cog_inertia_func(1-fuel_percentage_consumed)
    # thruster displacement from cog
    d_thrust_cg = d_thrust_cg_func(x_cog)

    # center of pressure
    d_cp_cg = CoP - x_cog

    # Angular dynamics
    thrust_moments_y = d_thrust_cg * thrust_x
    aero_moments_y = d_cp_cg * aero_x
    moments_y = thrust_moments_y + aero_moments_y    
    theta_dot_dot = moments_y / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = math.atan2(vy, vx)

    if theta > 2 * math.pi:
        theta -= 2 * math.pi

    alpha = theta - gamma

    mass -= mass_flow * dt

    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass]

    moments_dict = {
        'thrust_moments_y': thrust_moments_y,
        'aero_moements_y': aero_moments_y,
        'moments_y': moments_y,
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
        'x_cog': x_cog,
        'd_thrust_cg': d_thrust_cg
    }
    
    return state, propellant_mass, dynamic_pressure, info


def setup_physics_step_endo(dt,
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
    with open('data/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]

    with open('data/rocket_functions.pkl', 'rb') as f:  
        rocket_functions = dill.load(f)

    physics_step_lambda = lambda state, actions, propellant_mass: \
            rocket_model_physics_step_endo(state = state,
                                           actions = actions,
                                           propellant_mass = propellant_mass,
                                           dt = dt,
                                           initial_propellant_mass = sizing_results['Propellant mass stage 1 (ascent)'],
                                           cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                                           d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_0_lambda'],
                                           cop_func = rocket_functions['cop_subrocket_0_lambda'],
                                           frontal_area = sizing_results['Rocket frontal area'],
                                           v_exhaust = sizing_results['Exhaust velocity stage 1'],
                                           nozzle_exit_area = sizing_results['Nozzle exit area'],
                                           nozzle_exit_pressure = sizing_results['Nozzle exit pressure stage 1'],
                                           thrust_per_engine = sizing_results['Thrust engine stage 1'],
                                           number_of_engines_gimballed = sizing_results['Number of engines gimballed stage 1'],
                                           number_of_engines_non_gimballed = sizing_results['Number of engines stage 1'] - sizing_results['Number of engines gimballed stage 1'],
                                           CL_func = CL_func,
                                           CD_func = CD_func)
    return physics_step_lambda