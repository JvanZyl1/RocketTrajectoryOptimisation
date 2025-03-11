import math
import numpy as np
import csv
import dill

from src.envs.utils.atmosphere import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.Aero_coeffs import rocket_CL, rocket_CD

def triangle_wave(x: float):
    # Adjust the triangle wave to have the correct orientation
    if x > 0:
        return (1 - abs((x % 2) - 1))
    elif x == 0:
        return 1e-6
    else:
        return - (1 - abs((x % 2) - 1))

# Vertical rising and gravity turn
def rocket_model_physics_step_endo(state,
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
                      CD_func):
    
    # Clip actions at the physics level
    #action_scaling = 1e9
    #actions = actions / action_scaling
    actions = np.clip(actions, -1, 1)
    # x is through top of rocket, y is through side of rocket
    # x is unit force in x direction, u1 is throttle.
    u0, u1 = actions
    max_gimbal_angle_rad = math.radians(30)
    gimbal_angle_rad = max_gimbal_angle_rad * u0

    throttle = (u1 + 1) / 4 + 0.5 # [0-1]

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
    mass_flow = thrust_per_engine * (number_of_engines_gimballed + number_of_engines_non_gimballed) / v_exhaust

    thrust_engine_with_losses = (thrust_per_engine + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area) * throttle
    thrust_non_gimballed = thrust_engine_with_losses * number_of_engines_non_gimballed
    thrust_gimballed = thrust_engine_with_losses * number_of_engines_gimballed
    thrust_x = (thrust_non_gimballed + thrust_gimballed * math.cos(gimbal_angle_rad)) * math.cos(theta) + \
                thrust_gimballed * math.sin(gimbal_angle_rad) * math.sin(theta)
    thrust_y = (thrust_non_gimballed + thrust_gimballed * math.cos(gimbal_angle_rad)) * math.sin(theta) - \
                thrust_gimballed * math.sin(gimbal_angle_rad) * math.cos(theta)

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
    # thruster displacement from cog
    d_thrust_cg = d_thrust_cg_func(x_cog)

    # center of pressure
    d_cp_cg = CoP - x_cog

    # Angular dynamics
    thrust_moments_z = (thrust_x * math.sin(theta) - thrust_y * math.cos(theta)) * d_thrust_cg
    aero_moments_z = (-aero_x * math.sin(theta) + aero_y * math.cos(theta)) * d_cp_cg
    moments_z = thrust_moments_z + aero_moments_z 
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
        'thrust_moments_z': thrust_moments_z,
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
        'x_cog': x_cog,
        'd_thrust_cg': d_thrust_cg,
        'dynamic_pressure': dynamic_pressure,
        'gimbal_angle_deg': math.degrees(gimbal_angle_rad),
        'throttle': throttle
    }
    
    return state, info


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
            rocket_model_physics_step_endo(state = state,
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
                                           CD_func = CD_func)
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