import csv
import dill
import math
import numpy as np

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model, gravity_model_endo
from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD
from src.envs.disturbance_generator import VKDisturbanceGenerator

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
    if math.isnan(thrust_parallel):
        print(f'thrust parallel is nan. Non gimballed thrust is {thrust_non_gimballed}, thrust gimbaleld is {thrust_gimballed}'
              f'gimbal angle rad is {gimbal_angle_rad}'
              f'u0 is {u0} and {u1} is u1'
              f'atmospheric pressure {atmospheric_pressure}'
              f'd_thrust_cg {d_thrust_cg}')
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = math.degrees(gimbal_angle_rad)
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle

def first_order_low_pass_step(x, u, tau, dt):
    dx = (-x + u) / tau
    y = x + dt * dx
    return y

def force_moment_decomposer_flipoverboostbackburn(action,
                                      atmospheric_pressure,
                                      d_thrust_cg,
                                      gimbal_angle_deg_prev,
                                      dt,
                                      max_gimbal_angle_deg, # 45
                                      thrust_per_engine_no_losses,
                                      nozzle_exit_pressure,
                                      nozzle_exit_area,
                                      number_of_engines_flip_over_boostbackburn, # gimballed
                                      v_exhaust):
    gimbal_angle_command_deg = action * max_gimbal_angle_deg
    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = gimbal_angle_command_deg,
                                                 tau = 1.0,
                                                 dt = dt)
    gimbal_angle_rad = math.radians(gimbal_angle_deg)
    
    # No pressure losses but include for later graphs continuity.
    throttle = 1
    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_flip_over_boostbackburn * throttle

    thrust_parallel = thrust_gimballed * math.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total
    
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg

def ACS_single_deflection(deflection_command_deg,
        pitch_angle,
        flight_path_angle,
        dynamic_pressure,
        x_cog,
        delta_command_rad_prev,
        dt,
        grid_fin_area,
        CN_alpha,
        CN_0,
        CA_alpha,
        CA_0,
        d_base_grid_fin,
        zero_effective_aoa_test_case = False):
    # De-augment action wrt to coordinate frame, right up is positive, left down is positive
    max_deflection_angle_deg = 60
    delta_command_rad = math.radians(deflection_command_deg * max_deflection_angle_deg)

    # Pass through LPF
    delta_rad = first_order_low_pass_step(x = delta_command_rad_prev,
                                          u = delta_command_rad,
                                          tau = 0.5,
                                          dt = dt)
    alpha_effective_rad = flight_path_angle - pitch_angle - math.pi

    # local angle of attack
    alpha_local_rad = alpha_effective_rad + delta_rad

    # Normal and axial forces
    number_of_fins = 2
    Fn = number_of_fins * dynamic_pressure * grid_fin_area * (CN_alpha * alpha_local_rad + CN_0)
    Fa = number_of_fins * dynamic_pressure * grid_fin_area * (CA_alpha * alpha_local_rad + CA_0)
    Cn = CN_alpha * alpha_local_rad + CN_0
    Ca = CA_alpha * alpha_local_rad + CA_0

    # Forces
    force_parallel = Fa * math.sin(delta_rad) + Fn * math.cos(delta_rad)
    force_perpendicular = Fa * math.cos(delta_rad) + Fn * math.sin(delta_rad)

    # Moments
    Fx = force_parallel * math.cos(pitch_angle) + force_perpendicular * math.sin(pitch_angle)
    Fy = force_parallel * math.sin(pitch_angle) - force_perpendicular * math.cos(pitch_angle)
    d_fin_cg = abs(x_cog) - d_base_grid_fin
    moment_z = d_fin_cg * (-Fx * math.sin(pitch_angle) + Fy * math.cos(pitch_angle))

    if zero_effective_aoa_test_case:
        print(f'alpha_effective_rad : {alpha_effective_rad}, pitch_angle : {pitch_angle}, flight_path_angle : {flight_path_angle}')
        print(f'Cn : {Cn}, Cn_alpha : {CN_alpha}, Cn_0 : {CN_0}, alpha_local_rad : {alpha_local_rad}')
        print(f'Ca : {Ca}, Ca_alpha : {CA_alpha}, Ca_0 : {CA_0}, alpha_local_rad : {alpha_local_rad}')
        assert np.isclose(alpha_effective_rad, 0.0), "Alpha effective should be close to 0"
        assert np.isclose(Ca, 0.0), "Ca should be close to 0"
        assert np.isclose(alpha_local_rad, 0.0), "Alpha local should be close to 0"
        assert np.isclose(moment_z, 0.0), "Moment should be close to 0"
        assert force_parallel > 0, "Force parallel should be positive"
        assert np.isclose(force_perpendicular, 0.0), "Force perpendicular should be close to 0"
    
    return force_parallel, force_perpendicular, moment_z, delta_rad

def RCS(action,
        x_cog,
        max_RCS_force_per_thruster,
        d_base_rcs_bottom,
        d_base_rcs_top):
    thruster_force = max_RCS_force_per_thruster * action
    force_bottom = thruster_force * 20 # BEUN
    force_top = thruster_force * 20 # BEUN

    control_moment_z = (-force_bottom * (x_cog - d_base_rcs_bottom) + force_top * (d_base_rcs_top - x_cog))
    if type(control_moment_z) == np.float64 or type(control_moment_z) == float:
        pass
    else:
        control_moment_z = control_moment_z[0]
    control_force_parallel = 0
    control_force_perpendicular = 0
    mass_flow = 0
    return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow


def force_moment_decomposer_re_entry_burn(actions,
                                   atmospheric_pressure : float,
                                   d_thrust_cg : float,
                                   pitch_angle : float,
                                   flight_path_angle : float,
                                   dynamic_pressure : float,
                                   x_cog : float,
                                   delta_command_rad_prev : float,
                                   gimbal_angle_deg_prev : float,
                                   thrust_per_engine_no_losses : float,
                                   nozzle_exit_pressure : float,
                                   nozzle_exit_area : float,
                                   number_of_engines_gimballed : int, # All gimballed
                                   v_exhaust : float,
                                   grid_fin_area : float,
                                   CN_alpha : float,
                                   CN_0 : float,
                                   CA_alpha : float,
                                   CA_0 : float,
                                   d_base_grid_fin : float,
                                   nominal_throttle : float,
                                   dt : float,
                                   max_gimbal_angle_rad : float = math.radians(20)):
    # Actions : u0, u1
    # u0 is gimbal angle norm from -1 to 1
    # u1 is non nominal throttle from -1 to 1
    # u2 is deflection command norm from -1 to 1
    u0, u1 = actions
    u2 = 0
    gimbal_angle_rad = u0 * max_gimbal_angle_rad

    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = math.degrees(gimbal_angle_rad),
                                                 tau = 1.0,
                                                 dt = dt)
    gimbal_angle_rad = math.radians(gimbal_angle_deg)

    non_nominal_throttle = (u1 + 1) / 2

    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle
    thrust_non_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle

    thrust_parallel = thrust_non_gimballed + thrust_gimballed * math.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = math.degrees(gimbal_angle_rad)

    # ACS
    acs_force_parallel, acs_force_perpendicular, acs_moment_z, delta_rad = ACS(u2,
                                                                               pitch_angle,
                                                                               flight_path_angle,
                                                                               dynamic_pressure,
                                                                               x_cog,
                                                                               delta_command_rad_prev,
                                                                               dt,
                                                                               grid_fin_area,
                                                                               CN_alpha,
                                                                               CN_0,
                                                                               CA_alpha,
                                                                               CA_0,
                                                                               d_base_grid_fin)
    control_force_parallel = thrust_parallel + acs_force_parallel
    control_force_perpendicular = thrust_perpendicular + acs_force_perpendicular
    control_moment_z = moment_z + acs_moment_z
    return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad


def ACS(flight_path_angle : float,
        pitch_angle : float,
        dynamic_pressure_rel : float,
        x_cog : float,
        deflection_command_left_deg : float,
        deflection_command_right_deg : float,
        delta_command_left_rad_prev : float,
        delta_command_right_rad_prev : float,
        dt : float,
        C_N_0 : float,
        C_N_alpha : float,
        C_A_0 : float,
        C_A_alpha : float,
        grid_fin_area : float,
        d_base_grid_fin : float,
        rocket_radius : float):
    # De-augment action wrt to coordinate frame, right up is positive, left down is positive
    max_deflection_angle_deg = 60
    delta_command_left_rad = math.radians(deflection_command_left_deg * max_deflection_angle_deg)
    delta_command_right_rad = math.radians(deflection_command_right_deg * max_deflection_angle_deg)

    # Pass through LPF
    delta_left_rad = first_order_low_pass_step(x = delta_command_left_rad_prev,
                                          u = delta_command_left_rad,
                                          tau = 0.5,
                                          dt = dt)
    delta_right_rad = first_order_low_pass_step(x = delta_command_right_rad_prev,
                                          u = delta_command_right_rad,
                                          tau = 0.5,
                                          dt = dt)
    alpha_effective_rad = flight_path_angle - pitch_angle - math.pi

    # local angle of attack
    alpha_local_left_rad = alpha_effective_rad + delta_left_rad
    alpha_local_right_rad = alpha_effective_rad + delta_right_rad

    C_n_L = C_N_0 + C_N_alpha * abs(alpha_local_left_rad)
    C_a_L = C_A_0 + C_A_alpha * alpha_local_left_rad
    F_n_L = dynamic_pressure_rel * grid_fin_area * C_n_L
    F_a_L = dynamic_pressure_rel * grid_fin_area * C_a_L

    C_n_R = C_N_0 + C_N_alpha * abs(alpha_local_right_rad)
    C_a_R = C_A_0 + C_A_alpha * alpha_local_right_rad
    F_n_R = dynamic_pressure_rel * grid_fin_area * C_n_R
    F_a_R = dynamic_pressure_rel * grid_fin_area * C_a_R

    F_perpendicular_L = F_a_L * math.cos(delta_left_rad) - F_n_L * math.sin(delta_left_rad)
    F_perpendicular_R = F_a_R * math.cos(delta_right_rad) - F_n_R * math.sin(delta_right_rad)
    F_perpendicular = F_perpendicular_L + F_perpendicular_R


    # Assume perfect roll control, so 2 extra grid fins only enact a parallel force with equivalent of 0 alpha_effective_rad
    Fn_roll = 2 * dynamic_pressure_rel * grid_fin_area * C_N_0

    F_parallel_L = F_a_L * math.sin(delta_left_rad) + F_n_L * math.cos(delta_left_rad)
    F_parallel_R = F_a_R * math.sin(delta_right_rad) + F_n_R * math.cos(delta_right_rad)
    F_parallel = F_parallel_L + F_parallel_R + Fn_roll
    Fx = F_parallel * math.cos(pitch_angle) + F_perpendicular * math.sin(pitch_angle)
    Fy = F_parallel * math.sin(pitch_angle) - F_perpendicular * math.cos(pitch_angle)

    d_fin_cg = abs(d_base_grid_fin - x_cog) # d_base grid fin always further away, abs just in case worst case sceanairo, shouldn't be needed

    Mz = (F_perpendicular_R - F_perpendicular_L) * d_fin_cg + (F_parallel_R - F_parallel_L) * (rocket_radius+1)

    acs_info = {
        'alpha_local_left_rad' : alpha_local_left_rad,
        'alpha_local_right_rad' : alpha_local_right_rad,
        'C_n_L' : C_n_L,
        'C_a_L' : C_a_L,
        'C_n_R' : C_n_R,
        'C_a_R' : C_a_R,
        'F_n_L' : F_n_L,
        'F_a_L' : F_a_L,
        'F_n_R' : F_n_R,
        'F_a_R' : F_a_R,
        'F_perpendicular_L' : F_perpendicular_L,
        'F_perpendicular_R' : F_perpendicular_R,
        'F_perpendicular' : F_perpendicular,
        'F_parallel_L' : F_parallel_L,
        'F_parallel_R' : F_parallel_R,
        'F_parallel' : F_parallel,
        'Fx' : Fx,
        'Fy' : Fy,
        'Mz' : Mz,
        'd_fin_cg' : d_fin_cg,
        'delta_left_rad' : delta_left_rad,
        'delta_right_rad' : delta_right_rad        
    }

    return F_perpendicular, F_parallel, Mz, delta_command_left_rad, delta_command_right_rad, acs_info


def force_moment_decomposer_landing_burn(actions,
                                   atmospheric_pressure : float,
                                   d_thrust_cg : float,
                                   pitch_angle : float,
                                   flight_path_angle : float,
                                   dynamic_pressure_rel : float,
                                   x_cog : float,
                                   delta_command_left_rad_prev : float,
                                   delta_command_right_rad_prev : float,
                                   gimbal_angle_deg_prev : float,
                                   thrust_per_engine_no_losses : float,
                                   nozzle_exit_pressure : float,
                                   nozzle_exit_area : float,
                                   number_of_engines_gimballed : int, # All gimballed
                                   v_exhaust : float,
                                   grid_fin_area : float,
                                   CN_alpha : float,
                                   CN_0 : float,
                                   CA_alpha : float,
                                   CA_0 : float,
                                   d_base_grid_fin : float,
                                   nominal_throttle : float,
                                   dt : float,
                                   max_gimbal_angle_rad : float,
                                   max_deflection_angle_rad : float,
                                   rocket_radius : float):
    # Actions : u0, u1
    # u0 is gimbal angle norm from -1 to 1
    # u1 is non nominal throttle from -1 to 1
    # u2 is left deflection command norm from -1 to 1
    # u3 is right deflection command norm from -1 to 1
    # sac needs to squish actions as [[u0, u1, u2, u3]] whereas td3 is [u0, u1, u2, u3]
    # if not tuple
    if not isinstance(actions, tuple):
        if actions.ndim == 2:
            # Handle SAC format: [[u0, u1, u2, u3]]
            u0, u1, u2, u3 = actions[0]
        else:
            # Handle TD3 format: [u0, u1, u2, u3]
            u0, u1, u2, u3 = actions
    else:
        u0, u1, u2, u3 = actions
        
    gimbal_angle_rad = u0 * max_gimbal_angle_rad

    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = math.degrees(gimbal_angle_rad),
                                                 tau = 1.0,
                                                 dt = dt)
    gimbal_angle_rad = math.radians(gimbal_angle_deg)

    non_nominal_throttle = (u1 + 1) / 2

    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle
    thrust_non_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle

    thrust_parallel = thrust_non_gimballed + thrust_gimballed * math.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * math.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * math.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = np.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = math.degrees(gimbal_angle_rad)

    # ACS
    deflection_command_left_deg = u2 * max_deflection_angle_rad
    deflection_command_right_deg = u3 * max_deflection_angle_rad
    acs_force_perpendicular, acs_force_parallel, acs_moment_z, delta_command_left_rad, delta_command_right_rad, acs_info = \
        ACS(flight_path_angle = flight_path_angle,
            pitch_angle = pitch_angle,
            dynamic_pressure_rel = dynamic_pressure_rel,
            x_cog = x_cog,
            deflection_command_left_deg = deflection_command_left_deg,
            deflection_command_right_deg = deflection_command_right_deg,
            delta_command_left_rad_prev = delta_command_left_rad_prev,
            delta_command_right_rad_prev = delta_command_right_rad_prev,
            dt = dt,
            C_N_0 = CN_0,
            C_N_alpha = CN_alpha,
            C_A_0 = CA_0,
            C_A_alpha = CA_alpha,
            grid_fin_area = grid_fin_area,
            d_base_grid_fin = d_base_grid_fin,
            rocket_radius = rocket_radius)
    
    control_force_parallel = thrust_parallel + acs_force_parallel
    control_force_perpendicular = thrust_perpendicular + acs_force_perpendicular
    control_moment_z = moment_z + acs_moment_z
    return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_command_left_rad, delta_command_right_rad, acs_info
    

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
                      gimbal_angle_deg_prev : float = None,
                      delta_command_rad_prev : float = None,
                      delta_command_left_rad_prev : float = None,
                      delta_command_right_rad_prev : float = None,
                      wind_generator : VKDisturbanceGenerator = None):
    # Unpack state
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Atmopshere values
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
    speed = math.sqrt(vx**2 + vy**2)
    if speed_of_sound != 0.0:
        mach_number = speed / speed_of_sound
        # Max mach number logging
        Q_max = 30000 # [Pa]
        mach_number_logging = mach_number
        mach_number_max = math.sqrt(2 * Q_max / density) * 1 / speed_of_sound
    else:
        mach_number = 0.0 # IGNORE MACH NUMBER FOR NOW
        mach_number_logging = 200.0
        mach_number_max = 200.0
    dynamic_pressure = 0.5 * density * speed**2


    # Inertia and thrust displacement from cog
    fuel_percentage_consumed = (initial_propellant_mass_stage - mass_propellant) / initial_propellant_mass_stage
    if fuel_percentage_consumed == 0.0:
        fuel_percentage_consumed = 1e-6
    x_cog, inertia = cog_inertia_func(1-fuel_percentage_consumed)
    d_thrust_cg = d_thrust_cg_func(x_cog)

    # CoP and d_cp_cg
    if vy < 0:
        alpha_effective = gamma - theta - math.pi
    else:
        alpha_effective = alpha
    CoP = cop_func(math.degrees(alpha_effective), mach_number)
    d_cp_cg = x_cog - CoP


    # Get wind disturbance forces if generator is provided
    if wind_generator is not None:
        ug, vg = wind_generator(density)
    else:
        ug, vg = 0.0, 0.0

    # Determine later whether to do with Mach number of angle of attack
    if ug != 0.0 and vg != 0.0:
        vx_rel = vx - ug
        vy_rel = vy - vg
        speed_rel = math.sqrt(vx_rel**2 + vy_rel**2)
        gamma_rel = math.atan2(vy_rel, vx_rel)
        if gamma_rel < 0:
            gamma_rel = 2 * math.pi + gamma_rel
        if vy_rel < 0:
            alpha_effective_rel = gamma_rel - theta - math.pi
        else:
            alpha_effective_rel = theta - gamma_rel
    else:
        alpha_effective_rel = alpha_effective
        speed_rel = speed

    # Lift and drag
    C_L = CL_func(alpha_effective_rel, mach_number)
    C_D = CD_func(mach_number)
    drag = 0.5 * density * speed_rel**2 * C_D * frontal_area
    lift = 0.5 * density * speed_rel**2 * C_L * frontal_area
    if vy >= 0.0: 
        aero_force_parallel = lift * math.sin(alpha_effective_rel)  - drag * math.cos(alpha_effective_rel)
        aero_force_perpendicular = - lift * math.cos(alpha_effective_rel) - drag * math.sin(alpha_effective_rel)
    else:
        aero_force_parallel = drag * math.cos(alpha_effective_rel) - lift * math.sin(alpha_effective_rel)
        aero_force_perpendicular = - drag * math.sin(alpha_effective_rel) - lift * math.cos(alpha_effective_rel)
    aero_x = aero_force_parallel * math.cos(theta) + aero_force_perpendicular * math.sin(theta)
    aero_y = aero_force_parallel * math.sin(theta) - aero_force_perpendicular * math.cos(theta)
    aero_moments_z = aero_force_perpendicular * d_cp_cg
    dynamic_pressure_rel = 0.5 * density * speed_rel**2
    
    if flight_phase in ['subsonic', 'supersonic']:
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, \
             gimbal_angle_deg, throttle = control_function(actions, atmospheric_pressure, d_thrust_cg)
        action_info = {
            'gimbal_angle_deg': gimbal_angle_deg,
            'throttle': throttle
        }
    elif flight_phase == 'flip_over_boostbackburn':
        assert gimbal_angle_deg_prev is not None, "Gimbal angle degree previous is required for flip over"
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg = control_function(actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev)
        action_info = {
            'gimbal_angle_deg': gimbal_angle_deg
        }
    elif flight_phase == 'ballistic_arc_descent':
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow = control_function(actions, x_cog)
        action_info = {
            'RCS_throttle': actions
        }
    elif flight_phase == 're_entry_burn':
        assert gimbal_angle_deg_prev is not None, "Gimbal angle degree previous is required for re-entry burn"
        assert delta_command_rad_prev is not None, "Delta command rad previous is required for re-entry burn"
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad = control_function(actions, atmospheric_pressure, d_thrust_cg, theta, gamma, dynamic_pressure, x_cog, delta_command_rad_prev, gimbal_angle_deg_prev)
        action_info = {
            'throttle': throttle,
            'deflection_angle_rad': delta_rad,
            'gimbal_angle_deg': gimbal_angle_deg
        }
    elif flight_phase == 'landing_burn':
        assert delta_command_left_rad_prev is not None, "Previous left deflection command is required for landing burn"
        assert delta_command_right_rad_prev is not None, "Previous right deflection command is required for landing burn"
        assert gimbal_angle_deg_prev is not None, "Gimbal angle degree previous is required for re-entry burn" 
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_command_left_rad, delta_command_right_rad, acs_info = \
            control_function(actions, atmospheric_pressure, d_thrust_cg, theta, \
                                gamma, dynamic_pressure_rel, x_cog, delta_command_left_rad_prev, \
                                    delta_command_right_rad_prev, gimbal_angle_deg_prev)
        action_info = {
            'throttle': throttle,
            'delta_command_left_rad': delta_command_left_rad,
            'delta_command_right_rad': delta_command_right_rad,
            'gimbal_angle_deg': gimbal_angle_deg,
            'acs_info': acs_info
        }

    if math.isnan(control_force_parallel):
        print(f'Control force parallel is nan. State : {state}')
    elif math.isnan(control_force_perpendicular):
        print(f'Control force perpendicular is nan. State : {state}')
    elif math.isnan(control_moment_z):
        print(f'Moments are nan. State : {state}')
    
    control_force_x = (control_force_parallel) * math.cos(theta) + control_force_perpendicular * math.sin(theta)
    control_force_y = (control_force_parallel) * math.sin(theta) - control_force_perpendicular * math.cos(theta)

    # Gravity
    g = gravity_model_endo(y)

    # Forces
    forces_x = aero_x + control_force_x
    forces_y = aero_y + control_force_y

    # Kinematics
    vx_dot = forces_x/mass
    vy_dot = forces_y/mass - g
    vx += vx_dot * dt
    vy += vy_dot * dt
    x += vx * dt
    y += vy * dt

    # Angular dynamics
    moments_z = control_moment_z + aero_moments_z
    theta_dot_dot = moments_z / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = math.atan2(vy, vx)

    if theta > 2 * math.pi:
        theta -= 2 * math.pi
    if gamma < 0:
        gamma = 2 * math.pi + gamma

    alpha = theta - gamma

    # Mass Update
    mass_propellant -= mass_flow * dt
    mass -= mass_flow * dt

    time += dt

    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]


    acceleration_dict = {
        'acceleration_x_component_control': control_force_x/mass,
        'acceleration_y_component_control': control_force_y/mass,
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
        'control_moment_z': control_moment_z,
        'aero_moment_z': aero_moments_z,
        'moments_z': moments_z,
        'theta_dot_dot': theta_dot_dot
    }
    
    info = {
        'inertia': inertia,
        'acceleration_dict': acceleration_dict,
        'mach_number': mach_number_logging,
        'mach_number_max': mach_number_max,
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
        'control_force_parallel': control_force_parallel,
        'control_force_perpendicular': control_force_perpendicular,
        'control_force_x': control_force_x,
        'control_force_y': control_force_y,
        'aero_force_x': aero_x,
        'aero_force_y': aero_y,
        'gravity_force_y': -g*mass,
        'atmospheric_pressure': atmospheric_pressure,
        'air_density': density,
        'speed_of_sound': speed_of_sound,
        'action_info': action_info,
        'ug': ug,
        'vg': vg
    }

    return state, info


def compile_physics(dt,
                    flight_phase : str):

    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn', 'landing_burn']
    CL_func = lambda alpha, M: rocket_CL(alpha, M)
    CD_func = lambda M: rocket_CD(M)

    # Read sizing results
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]

    with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
        rocket_functions = dill.load(f)

    cop_func_full_rocket_ascent = lambda alpha, M: rocket_functions['cop_subrocket_0_lambda'](alpha, M)
    cop_func_stage_2_ascent = lambda alpha, M: rocket_functions['cop_subrocket_1_lambda'](alpha, M)
    cop_func_stage_1_descent = lambda alpha, M: rocket_functions['cop_subrocket_2_lambda'](alpha, M)

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
                                           max_gimbal_angle_rad = math.radians(0.5))

        physics_step_lambda = lambda state, actions, wind_generator: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = float(sizing_results['Actual propellant mass stage 1'])*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_0_lambda'],
                                   cop_func = cop_func_full_rocket_ascent,
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   wind_generator = wind_generator)
    elif flight_phase == 'flip_over_boostbackburn':
        force_composer_lambda = lambda actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev : \
            force_moment_decomposer_flipoverboostbackburn(actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev,
                                              dt = dt,
                                              max_gimbal_angle_deg=45,
                                              thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
                                              nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                                              nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                              number_of_engines_flip_over_boostbackburn = 6,
                                              v_exhaust = float(sizing_results['Exhaust velocity stage 1']))
        physics_step_lambda = lambda state, actions, gimbal_angle_deg_prev, wind_generator: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = float(sizing_results['Actual propellant mass stage 1'])*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda'],
                                   cop_func = cop_func_stage_1_descent,
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   gimbal_angle_deg_prev = gimbal_angle_deg_prev,
                                   wind_generator = wind_generator)
    elif flight_phase == 'ballistic_arc_descent':
        force_composer_lambda = lambda action, x_cog : \
            RCS(action,
                x_cog,
                float(sizing_results['max_RCS_force_per_thruster']),
                float(sizing_results['d_base_rcs_bottom']),
                float(sizing_results['d_base_rcs_top']))
        physics_step_lambda = lambda state, actions, wind_generator: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = float(sizing_results['Actual propellant mass stage 1'])*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda'],
                                   cop_func = cop_func_stage_1_descent,
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   gimbal_angle_deg_prev = None,
                                   delta_command_rad_prev = None,
                                   wind_generator = wind_generator)
    elif flight_phase == 're_entry_burn':
        number_of_engines_min = 3
        minimum_engine_throttle = 0.4
        nominal_throttle_re_entry_burn = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
        force_composer_lambda = lambda actions, atmospheric_pressure, d_thrust_cg, pitch_angle, flight_path_angle, dynamic_pressure, x_cog, delta_command_rad_prev, gimbal_angle_deg_prev : \
                        force_moment_decomposer_re_entry_burn(actions = actions,
                                                                      atmospheric_pressure = atmospheric_pressure,
                                                                      d_thrust_cg = d_thrust_cg,
                                                                      pitch_angle = pitch_angle,
                                                                      flight_path_angle = flight_path_angle,
                                                                      dynamic_pressure = dynamic_pressure,
                                                                      x_cog = x_cog,
                                                                      delta_command_rad_prev = delta_command_rad_prev,
                                                                      gimbal_angle_deg_prev = gimbal_angle_deg_prev,
                                                                      thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
                                                                      nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
                                                                      nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                                                      number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1']), # All gimballed
                                                                      v_exhaust = float(sizing_results['Exhaust velocity stage 1']),
                                                                      grid_fin_area = float(sizing_results['S_grid_fins']),
                                                                      CN_alpha = float(sizing_results['C_n_alpha_local']),
                                                                      CN_0 = float(sizing_results['C_n_0']),
                                                                      CA_alpha = float(sizing_results['C_a_alpha_local']),
                                                                      CA_0 = float(sizing_results['C_a_0']),
                                                                      d_base_grid_fin = float(sizing_results['d_base_grid_fin']),
                                                                      nominal_throttle = nominal_throttle_re_entry_burn,
                                                                      dt = dt,
                                                                      max_gimbal_angle_rad = math.radians(2))
        
        physics_step_lambda = lambda state, actions, gimbal_angle_deg_prev, delta_command_rad_prev, wind_generator: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = float(sizing_results['Actual propellant mass stage 1'])*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda'],
                                   cop_func = cop_func_stage_1_descent,
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   gimbal_angle_deg_prev = gimbal_angle_deg_prev,
                                   delta_command_rad_prev = delta_command_rad_prev,
                                   wind_generator = wind_generator)
    elif flight_phase == 'landing_burn':
        number_of_engines_min = 3
        minimum_engine_throttle = 0.4
        nominal_throttle_re_entry_burn = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
        force_composer_lambda = lambda actions, atmospheric_pressure, d_thrust_cg, pitch_angle, \
                    flight_path_angle, dynamic_pressure_rel, x_cog, delta_command_left_rad_prev, \
                        delta_command_right_rad_prev, gimbal_angle_deg_prev : \
                            force_moment_decomposer_landing_burn(actions,
                                                                 atmospheric_pressure,
                                                                 d_thrust_cg,
                                                                 pitch_angle,
                                                                 flight_path_angle,
                                                                 dynamic_pressure_rel,
                                                                 x_cog,
                                                                 delta_command_left_rad_prev,
                                                                 delta_command_right_rad_prev,
                                                                 gimbal_angle_deg_prev,
                                                                 thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
                                                                 nozzle_exit_pressure  = float(sizing_results['Nozzle exit pressure stage 1']),
                                                                 nozzle_exit_area = float(sizing_results['Nozzle exit area']),
                                                                 number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1']), # All gimballed
                                                                 v_exhaust = float(sizing_results['Exhaust velocity stage 1']),
                                                                 grid_fin_area = float(sizing_results['S_grid_fins']),
                                                                 CN_alpha  = float(sizing_results['C_n_alpha_local']),
                                                                 CN_0 = float(sizing_results['C_n_0']),
                                                                 CA_alpha = float(sizing_results['C_a_alpha_local']),
                                                                 CA_0 = float(sizing_results['C_a_0']),
                                                                 d_base_grid_fin = float(sizing_results['d_base_grid_fin']),
                                                                 nominal_throttle = nominal_throttle_re_entry_burn,
                                                                 dt = dt,
                                                                 max_gimbal_angle_rad = math.radians(1),
                                                                 max_deflection_angle_rad = math.radians(20),
                                                                 rocket_radius = float(sizing_results['Rocket Radius']))
        
        physics_step_lambda = lambda state, actions, gimbal_angle_deg_prev, delta_command_left_rad_prev, delta_command_right_rad_prev, wind_generator: \
                rocket_physics_fcn(state = state,
                                   actions = actions,
                                   dt = dt,
                                   flight_phase = flight_phase,
                                   control_function = force_composer_lambda,
                                   initial_propellant_mass_stage = float(sizing_results['Actual propellant mass stage 1'])*1000,
                                   cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda'],
                                   d_thrust_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda'],
                                   cop_func = cop_func_stage_1_descent,
                                   frontal_area = float(sizing_results['Rocket frontal area']),
                                   CL_func = CL_func,
                                   CD_func = CD_func,
                                   gimbal_angle_deg_prev = gimbal_angle_deg_prev,
                                   delta_command_left_rad_prev = delta_command_left_rad_prev,
                                   delta_command_right_rad_prev = delta_command_right_rad_prev,
                                   wind_generator = wind_generator)
    return physics_step_lambda