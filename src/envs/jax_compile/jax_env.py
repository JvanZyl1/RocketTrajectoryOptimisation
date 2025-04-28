import jax
import csv
import dill
from functools import partial
import jax.numpy as jnp

@partial(jax.jit,
         static_argnames = ['thrust_per_engine_no_losses',
                            'nozzle_exit_pressure',
                            'nozzle_exit_area',
                            'number_of_engines_gimballed',
                                'number_of_engines_non_gimballed',
                                'v_exhaust',
                                'nominal_throttle',
                                'max_gimbal_angle_rad'])
def force_moment_decomposer_ascent_jax(actions : jnp.ndarray,
                                   atmospheric_pressure : float,
                                   d_thrust_cg : float,
                                   thrust_per_engine_no_losses : float,
                                   nozzle_exit_pressure : float,
                                   nozzle_exit_area : float,
                                   number_of_engines_gimballed : int,
                                   number_of_engines_non_gimballed : int,
                                   v_exhaust : float,
                                   nominal_throttle : float = 0.5,
                                   max_gimbal_angle_rad : float = jnp.radians(1)):
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

    thrust_parallel = thrust_non_gimballed + thrust_gimballed * jnp.cos(gimbal_angle_rad)
    if jnp.isnan(thrust_parallel):
        print(f'thrust parallel is nan. Non gimballed thrust is {thrust_non_gimballed}, thrust gimbaleld is {thrust_gimballed}'
              f'gimbal angle rad is {gimbal_angle_rad}'
              f'u0 is {u0} and {u1} is u1'
              f'atmospheric pressure {atmospheric_pressure}'
              f'd_thrust_cg {d_thrust_cg}')
    thrust_perpendicular = - thrust_gimballed * jnp.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * jnp.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = jnp.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = jnp.degrees(gimbal_angle_rad)
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle

def first_order_low_pass_step(x, u, tau, dt):
    dx = (-x + u) / tau
    y = x + dt * dx
    return y

@partial(jax.jit,
         static_argnames = ['dt',
                            'max_gimbal_angle_deg',
                            'thrust_per_engine_no_losses',
                            'nozzle_exit_pressure',
                            'nozzle_exit_area',
                            'number_of_engines_flip_over_boostbackburn',
                            'v_exhaust'])
def force_moment_decomposer_flipoverboostbackburn_jax(action : jnp.ndarray,
                                      atmospheric_pressure : float,
                                      d_thrust_cg : float,
                                      gimbal_angle_deg_prev : float,
                                      dt : float,
                                      max_gimbal_angle_deg : float, # 45
                                      thrust_per_engine_no_losses : float,
                                      nozzle_exit_pressure : float,
                                      nozzle_exit_area : float,
                                      number_of_engines_flip_over_boostbackburn : int, # gimballed
                                      v_exhaust : float):
    gimbal_angle_command_deg = action * max_gimbal_angle_deg
    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = gimbal_angle_command_deg,
                                                 tau = 1.0,
                                                 dt = dt)
    gimbal_angle_rad = jnp.radians(gimbal_angle_deg)
    
    # No pressure losses but include for later graphs continuity.
    throttle = 1
    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_flip_over_boostbackburn * throttle

    thrust_parallel = thrust_gimballed * jnp.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * jnp.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * jnp.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = jnp.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = jnp.degrees(gimbal_angle_rad)
    return thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle

@partial(jax.jit,
         static_argnames = ['dt',
                            'grid_fin_area',
                            'CN_alpha',
                            'CN_0',
                            'CA_alpha',
                            'CA_0',
                            'd_base_grid_fin'])
def ACS_jax(deflection_command_deg : float,
            pitch_angle : float,
            flight_path_angle : float,
            dynamic_pressure : float,
            x_cog : float,
            delta_command_rad_prev : float,
            dt : float,
            grid_fin_area : float,
            CN_alpha : float,
            CN_0 : float,
            CA_alpha : float,
            CA_0 : float,
            d_base_grid_fin : float):
    # De-augment action wrt to coordinate frame, right up is positive, left down is positive
    max_deflection_angle_deg = 60
    delta_command_rad = jnp.radians(deflection_command_deg * max_deflection_angle_deg)

    # Pass through LPF
    delta_rad = first_order_low_pass_step(x = delta_command_rad_prev,
                                          u = delta_command_rad,
                                          tau = 0.5,
                                          dt = dt)
    alpha_effective_rad = pitch_angle - flight_path_angle - jnp.pi

    # local angle of attack
    alpha_local_rad = alpha_effective_rad + delta_rad

    # Normal and axial forces
    number_of_fins = 2
    Fn = number_of_fins * dynamic_pressure * grid_fin_area * (CN_alpha * alpha_local_rad + CN_0)
    Fa = number_of_fins * dynamic_pressure * grid_fin_area * (CA_alpha * alpha_local_rad + CA_0)

    # Forces
    force_parallel = Fa * jnp.sin(delta_rad) + Fn * jnp.cos(delta_rad)
    force_perpendicular = Fa * jnp.cos(delta_rad) + Fn * jnp.sin(delta_rad)

    # Moments
    Fx = force_parallel * jnp.cos(pitch_angle) + force_perpendicular * jnp.sin(pitch_angle)
    Fy = force_parallel * jnp.sin(pitch_angle) - force_perpendicular * jnp.cos(pitch_angle)
    d_fin_cg = jnp.abs(x_cog) - d_base_grid_fin
    moment_z = d_fin_cg * (-Fx * jnp.sin(pitch_angle) + Fy * jnp.cos(pitch_angle))
    
    return force_parallel, force_perpendicular, moment_z, delta_rad

@partial(jax.jit,
         static_argnames = ['max_RCS_force_per_thruster',
                            'd_base_rcs_bottom',
                            'd_base_rcs_top'])
def RCS_jax(action : jnp.ndarray,
        x_cog : float,
        max_RCS_force_per_thruster : float,
        d_base_rcs_bottom : float,
        d_base_rcs_top : float):
    thruster_force = max_RCS_force_per_thruster * action
    force_bottom = thruster_force * 60 # BEUN
    force_top = thruster_force * 60 # BEUN

    control_moment_z = (-force_bottom * (x_cog - d_base_rcs_bottom) + force_top * (d_base_rcs_top - x_cog))
    control_force_parallel = 0
    control_force_perpendicular = 0
    mass_flow = 0
    return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow



@partial(jax.jit,
         static_argnames = ['thrust_per_engine_no_losses',
                            'nozzle_exit_pressure',
                            'nozzle_exit_area',
                            'number_of_engines_gimballed',
                            'v_exhaust',
                            'grid_fin_area',
                            'CN_alpha',
                            'CN_0',
                            'CA_alpha',
                            'CA_0',
                            'd_base_grid_fin',
                            'nominal_throttle',
                            'dt',
                            'max_gimbal_angle_rad'])
def force_moment_decomposer_re_entry_landing_burn_jax(actions : jnp.ndarray,
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
                                   max_gimbal_angle_rad : float = jnp.radians(20)):
    # Actions : u0, u1
    # u0 is gimbal angle norm from -1 to 1
    # u1 is non nominal throttle from -1 to 1
    # u2 is deflection command norm from -1 to 1
    u0, u1 = actions
    u2 = 0
    gimbal_angle_rad = u0 * max_gimbal_angle_rad

    gimbal_angle_deg = first_order_low_pass_step(x = gimbal_angle_deg_prev,
                                                 u = jnp.degrees(gimbal_angle_rad),
                                                 tau = 1.0,
                                                 dt = dt)
    gimbal_angle_rad = jnp.radians(gimbal_angle_deg)

    non_nominal_throttle = (u1 + 1) / 2

    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle
    thrust_non_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle

    thrust_parallel = thrust_non_gimballed + thrust_gimballed * jnp.cos(gimbal_angle_rad)
    thrust_perpendicular = - thrust_gimballed * jnp.sin(gimbal_angle_rad)
    moment_z = - thrust_gimballed * jnp.sin(gimbal_angle_rad) * d_thrust_cg

    total_thrust = jnp.sqrt(thrust_parallel**2 + thrust_perpendicular**2)
    number_of_engines_thrust_total = total_thrust / thrust_engine_with_losses_full_throttle
    mass_flow = (thrust_per_engine_no_losses / v_exhaust) * number_of_engines_thrust_total

    gimbal_angle_deg = jnp.degrees(gimbal_angle_rad)

    # ACS
    acs_force_parallel, acs_force_perpendicular, acs_moment_z, delta_rad = ACS_jax(deflection_command_deg = u2,
                                                                               pitch_angle = pitch_angle,
                                                                               flight_path_angle = flight_path_angle,
                                                                               dynamic_pressure = dynamic_pressure,
                                                                               x_cog = x_cog,
                                                                               delta_command_rad_prev = delta_command_rad_prev,
                                                                               dt = dt,
                                                                               grid_fin_area = grid_fin_area,
                                                                               CN_alpha = CN_alpha,
                                                                               CN_0 = CN_0,
                                                                               CA_alpha = CA_alpha,
                                                                               CA_0 = CA_0,
                                                                               d_base_grid_fin = d_base_grid_fin)
    control_force_parallel = thrust_parallel + acs_force_parallel
    control_force_perpendicular = thrust_perpendicular + acs_force_perpendicular
    control_moment_z = moment_z + acs_moment_z
    return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad

def rocket_CL(alpha,                    # [rad]
              M,                        # [-]
              kl_sub = 2.0,             # effective lift slope in subsonic flight for a typical rocket
              kl_sup = 1.0              # reduced lift slope in supersonic flight
              ): # radians & -
    """
    For a rocket, the overall normal force coefficient derivative is lower than the thin-airfoil value.
    Here we assume:
      - Subsonic (M < 0.8): effective lift slope circa 2.0 per radian, with Prandtl-Glauert compressibility correction.
      - Transonic (0.8 leq M geq 1.2): linear interpolation between subsonic and supersonic slopes.
      - Supersonic (M > 1.2): reduced lift slope circa 1.0 per radian.
    """
    
    def subsonic_case():
        comp_factor = 1.0 / jnp.sqrt(1 - M**2)
        return kl_sub * alpha * comp_factor
    
    def transonic_case():
        t = (M - 0.8) / 0.4
        # Evaluate subsonic value at M = 0.8
        comp_sub = 1.0 / jnp.sqrt(1 - 0.8**2)
        sub_val = kl_sub * alpha * comp_sub
        sup_val = kl_sup * alpha
        return (1 - t) * sub_val + t * sup_val
    
    def supersonic_case():
        return kl_sup * alpha
    
    return jax.lax.cond(
        M < 0.8,
        subsonic_case,
        lambda: jax.lax.cond(
            M <= 1.2,
            transonic_case,
            supersonic_case
        )
    )

def rocket_CD(alpha,                # [rad]
              M,                    # [-]
              cd0_subsonic=0.05,    # zero-lift drag coefficient in subsonic flight
              kd_subsonic=0.5,      # induced drag scaling in subsonic flight
              cd0_supersonic=0.10, # zero-lift drag coefficient in supersonic flight
              kd_supersonic=1.0    # induced drag scaling in supersonic flight
              ):
    """
    For a rocket, the drag is composed of:
      - A baseline zero-lift drag (cd0) that accounts for body, fin, and wave drag effects.
      - An induced drag term that scales roughly as α².
    We assume:
      - Subsonic (M < 0.8): cd0_subsonic circa 0.05 (with compressibility correction) and induced drag scaling kd_subsonic circa 0.5.
      - Transonic (0.8 leq M geq 1.2): linear interpolation between subsonic and supersonic parameters.
      - Supersonic (M > 1.2): cd0_supersonic circa 0.10 and induced drag scaling kd_supersonic circa 1.0.
    """
    def subsonic_case():
        comp_factor = 1.0 / jnp.sqrt(1 - M**2)
        return cd0_subsonic * comp_factor + kd_subsonic * (alpha**2)
    
    def transonic_case():
        t = (M - 0.8) / 0.4
        comp_sub = 1.0 / jnp.sqrt(1 - 0.8**2)
        sub_val = cd0_subsonic * comp_sub + kd_subsonic * (alpha**2)
        sup_val = cd0_supersonic + kd_supersonic * (alpha**2)
        return (1 - t) * sub_val + t * sup_val
    
    def supersonic_case():
        return cd0_supersonic + kd_supersonic * (alpha**2)
    
    return jax.lax.cond(
        M < 0.8,
        subsonic_case,
        lambda: jax.lax.cond(
            M <= 1.2,
            transonic_case,
            supersonic_case
        )
    )


def rocket_physics_fcn(state : jnp.array,
                      actions : jnp.array,
                      # Lambda wrapped
                      flight_phase : str,
                      control_function : callable,
                      dt : float,
                      initial_propellant_mass_stage : float,
                      cog_inertia_func_jitted : callable,
                      d_thrust_cg_func_jitted : callable,
                      cop_func : callable,
                      frontal_area : float,
                      CL_func : callable,
                      CD_func : callable,
                      gimbal_angle_deg_prev : float = None,
                      delta_command_rad_prev : float = None):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    fuel_percentage_consumed = (initial_propellant_mass_stage - mass_propellant) / initial_propellant_mass_stage
    fuel_percentage_consumed = jax.lax.cond(
        fuel_percentage_consumed == 0.0,
        lambda: 1e-6,
        lambda: fuel_percentage_consumed
    )
    x_cog, inertia = cog_inertia_func_jitted(1-fuel_percentage_consumed)
    d_thrust_cg = d_thrust_cg_func_jitted(x_cog)

    # Atmopshere values
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
    speed = jnp.sqrt(vx**2 + vy**2)
    max_number_max = jax.lax.cond(
        speed_of_sound != 0.0,
        lambda: jnp.sqrt(2 * 30000 / density) * 1 / speed_of_sound,
        lambda: 200.0)
    mach_number = jax.lax.cond(
        speed_of_sound != 0.0,
        lambda: speed / speed_of_sound,
        lambda: 0.0
    )
    dynamic_pressure = 0.5 * density * speed**2

    control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad = \
        control_function(actions, atmospheric_pressure, d_thrust_cg,
                                       gimbal_angle_deg_prev, x_cog, theta,
                                       gamma, dynamic_pressure, delta_command_rad_prev)
    control_force_x = (control_force_parallel) * jnp.cos(theta) + control_force_perpendicular * jnp.sin(theta)
    control_force_y = (control_force_parallel) * jnp.sin(theta) - control_force_perpendicular * jnp.cos(theta)

    # Gravity
    g = gravity_model_endo(y)

    # Determine later whether to do with Mach number of angle of attack
    alpha_effective = jax.lax.cond(
        vy < 0,
        lambda: gamma - theta - jnp.pi,
        lambda: alpha
    )
    C_L = CL_func(alpha_effective, mach_number)
    C_D = CD_func(alpha_effective, mach_number)
    CoP = cop_func(jnp.degrees(alpha_effective), mach_number)
    d_cp_cg = CoP - x_cog

    # Lift and drag
    drag = 0.5 * density * speed**2 * C_D * frontal_area
    lift = 0.5 * density * speed**2 * C_L * frontal_area
    aero_x = -drag * jnp.cos(gamma) - lift * jnp.cos(jnp.pi - gamma)
    aero_y = -drag * jnp.sin(gamma) + lift * jnp.sin(jnp.pi - gamma)

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
    aero_moments_z = (-aero_x * jnp.sin(theta) + aero_y * jnp.cos(theta)) * d_cp_cg
    moments_z = control_moment_z + aero_moments_z 
    theta_dot_dot = moments_z / inertia
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    gamma = jnp.atan2(vy, vx)

    if theta > 2 * jnp.pi:
        theta -= 2 * jnp.pi
    if gamma < 0:
        gamma = 2 * jnp.pi + gamma

    alpha = theta - gamma

    # Mass Update
    mass_propellant -= mass_flow * dt
    mass -= mass_flow * dt

    time += dt

    state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]


    acceleration_dict = {
        'acceleration_x_component_control': control_force_x/mass,
        'acceleration_y_component_control': control_force_y/mass,
        'acceleration_x_component_drag': -drag * jnp.cos(gamma)/mass,
        'acceleration_y_component_drag': -drag/mass * jnp.sin(gamma)/mass,
        'acceleration_x_component_lift': - lift * jnp.cos(jnp.pi - gamma)/mass,
        'acceleration_y_component_lift': lift * jnp.sin(jnp.pi - gamma)/mass,
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
        'action_info': action_info
    }

    return state, info



class JAX_Simulator:
    def __init__(self,
                 flight_phase : str,
                 kl_sub = 2.0,
                 kl_sup = 1.0,
                 cd0_subsonic=0.05,
                 kd_subsonic=0.5,
                 cd0_supersonic=0.10,
                 kd_supersonic=1.0):
        self.flight_phase = flight_phase
        self.dt = 0.1

        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
        self.CL_func_jit = jax.jit(
             partial(rocket_CL,
                     kl_sub = kl_sub,
                     kl_sup = kl_sup),
             static_argnames = ['kl_sub', 'kl_sup']
        )
        self.CD_func_jit = jax.jit(
             partial(rocket_CD,
                     cd0_subsonic = cd0_subsonic,
                     kd_subsonic = kd_subsonic,
                     cd0_supersonic = cd0_supersonic,
                     kd_supersonic = kd_supersonic),
             static_argnames = ['cd0_subsonic', 'kd_subsonic', 'cd0_supersonic', 'kd_supersonic']
        )

        # Read sizing results
        self.sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.sizing_results[row[0]] = row[2]

        with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
            self.rocket_functions = dill.load(f)

    def control_action_jax_compiler(self):
        if self.flight_phase in ['subsonic', 'supersonic']:
            force_composer_ascent_jit = jax.jit(
                partial(force_moment_decomposer_ascent_jax,
                        thrust_per_engine_no_losses = float(self.sizing_results['Thrust engine stage 1']),
                        nozzle_exit_pressure = float(self.sizing_results['Nozzle exit pressure stage 1']),
                        nozzle_exit_area = float(self.sizing_results['Nozzle exit area']),
                        number_of_engines_gimballed = int(self.sizing_results['Number of engines gimballed stage 1']),
                        number_of_engines_non_gimballed = int(self.sizing_results['Number of engines stage 1']) \
                            - int(self.sizing_results['Number of engines gimballed stage 1']),
                        v_exhaust = float(self.sizing_results['Exhaust velocity stage 1']),
                        nominal_throttle = 0.5,
                        max_gimbal_angle_rad = jnp.radians(1)),
                static_argnames = ['thrust_per_engine_no_losses', 'nozzle_exit_pressure', \
                                   'nozzle_exit_area', 'number_of_engines_gimballed', 'number_of_engines_non_gimballed', \
                                    'v_exhaust', 'nominal_throttle', 'max_gimbal_angle_rad']
            )
            def control_func_inference(actions, atmospheric_pressure, d_thrust_cg,
                                       gimbal_angle_deg_prev, x_cog, pitch_angle,
                                       flight_path_angle, dynamic_pressure, delta_command_rad_prev):
                control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle = \
                    force_composer_ascent_jit(actions, atmospheric_pressure, d_thrust_cg)
                delta_rad = 0.0
                return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad

        elif self.flight_phase == 'flip_over_boostbackburn':
            force_composer_flipoverboostbackburn_jit = jax.jit(
                partial(force_moment_decomposer_flipoverboostbackburn_jax,
                        thrust_per_engine_no_losses = float(self.sizing_results['Thrust engine stage 1']),
                        nozzle_exit_pressure = float(self.sizing_results['Nozzle exit pressure stage 1']),
                        nozzle_exit_area = float(self.sizing_results['Nozzle exit area']),
                        number_of_engines_flip_over_boostbackburn = 6,
                        v_exhaust = float(self.sizing_results['Exhaust velocity stage 1'])),
                static_argnames = ['thrust_per_engine_no_losses', 'nozzle_exit_pressure', \
                                   'nozzle_exit_area', 'number_of_engines_flip_over_boostbackburn', 'v_exhaust']
            )
            def control_func_inference(actions, atmospheric_pressure, d_thrust_cg,
                                       gimbal_angle_deg_prev, x_cog, pitch_angle,
                                       flight_path_angle, dynamic_pressure, delta_command_rad_prev):
                control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle = \
                    force_composer_flipoverboostbackburn_jit(actions, atmospheric_pressure, d_thrust_cg, gimbal_angle_deg_prev)
                delta_rad = 0.0
                return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad
            
        elif self.flight_phase == 'ballistic_arc_descent':
            force_composer_ballistic_arc_descent_jit = jax.jit(
                partial(RCS_jax,
                        max_RCS_force_per_thruster = float(self.sizing_results['max_RCS_force_per_thruster']),
                        d_base_rcs_bottom = float(self.sizing_results['d_base_rcs_bottom']),
                        d_base_rcs_top = float(self.sizing_results['d_base_rcs_top'])),
                static_argnames = ['x_cog', 'max_RCS_force_per_thruster', 'd_base_rcs_bottom', 'd_base_rcs_top']
            )
            def control_func_inference(actions, atmospheric_pressure, d_thrust_cg,
                                       gimbal_angle_deg_prev, x_cog, pitch_angle,
                                       flight_path_angle, dynamic_pressure, delta_command_rad_prev):
                control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow = \
                    force_composer_ballistic_arc_descent_jit(actions, x_cog)
                delta_rad = 0.0
                gimbal_angle_deg = 0.0
                throttle = 0.0
                return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad

        elif self.flight_phase == 're_entry_burn':
            number_of_engines_min = 3
            minimum_engine_throttle = 0.4
            nominal_throttle_re_entry_burn = (number_of_engines_min * minimum_engine_throttle) / int(self.sizing_results['Number of engines gimballed stage 1'])
            force_composer_re_entry_burn_jit = jax.jit(
                partial(force_moment_decomposer_re_entry_landing_burn_jax,
                        thrust_per_engine_no_losses = float(self.sizing_results['Thrust engine stage 1']),
                        nozzle_exit_pressure = float(self.sizing_results['Nozzle exit pressure stage 1']),
                        nozzle_exit_area = float(self.sizing_results['Nozzle exit area']),
                        number_of_engines_gimballed = int(self.sizing_results['Number of engines gimballed stage 1']), # All gimballed
                        v_exhaust = float(self.sizing_results['Exhaust velocity stage 1']),
                        grid_fin_area = float(self.sizing_results['S_grid_fins']),
                        CN_alpha = float(self.sizing_results['C_n_alpha_local']),
                        CN_0 = float(self.sizing_results['C_n_0']),
                        CA_alpha = float(self.sizing_results['C_a_alpha_local']),
                        CA_0 = float(self.sizing_results['C_a_0']),
                        d_base_grid_fin = float(self.sizing_results['d_base_grid_fin']),
                        nominal_throttle = nominal_throttle_re_entry_burn,
                        dt = self.dt,
                        max_gimbal_angle_rad = jnp.radians(20)),
                static_argnames = ['thrust_per_engine_no_losses', 'nozzle_exit_pressure', \
                                   'nozzle_exit_area', 'number_of_engines_gimballed', 'v_exhaust', \
                                    'grid_fin_area', 'CN_alpha', 'CN_0', 'CA_alpha', 'CA_0', 'd_base_grid_fin', 'nominal_throttle', 'dt', 'max_gimbal_angle_rad']
            )
            def control_func_inference(actions, atmospheric_pressure, d_thrust_cg,
                                       gimbal_angle_deg_prev, x_cog, pitch_angle,
                                       flight_path_angle, dynamic_pressure, delta_command_rad_prev):
                control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad = \
                force_composer_re_entry_burn_jit(actions, atmospheric_pressure, d_thrust_cg, \
                                                        pitch_angle, flight_path_angle, dynamic_pressure, \
                                                              x_cog, delta_command_rad_prev, gimbal_angle_deg_prev)
                return control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad
        return control_func_inference