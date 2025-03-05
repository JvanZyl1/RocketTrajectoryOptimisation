import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from TrajectoryGeneration.constraints import return_constraints
from TrajectoryGeneration.cost_function import cost_fcn
from TrajectoryGeneration.final_mass_compute import final_mass_compute

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def exo_dyn(t,
            augmented_state_vector,
            v_exhaust_stage_2,
            mass_flow_per_engine_stage_2,
            number_of_engines_stage_2):
    """
    Defines the ODE system.
    x = [r_x, r_y, r_z, v_x, v_y, v_z, m, p_r_x, p_r_y, p_r_z, p_v_x, p_v_y, p_v_z]
    """
    thrust = v_exhaust_stage_2 * mass_flow_per_engine_stage_2 * number_of_engines_stage_2
    r = augmented_state_vector[0:3]
    v = augmented_state_vector[3:6]
    m = augmented_state_vector[6]
    p_r = augmented_state_vector[7:10]
    p_v = augmented_state_vector[10:13]

    r_dot = v
    v_dot = (-mu / np.linalg.norm(r)**3) * r + (thrust / m) * (p_v / np.linalg.norm(p_v))
    pr_dot = - (mu / np.linalg.norm(r)**3) * (3 * np.dot(p_v, r) * r / np.linalg.norm(r)**2 - p_v)
    pv_dot = -p_r

    m_dot = -mass_flow_per_engine_stage_2 * number_of_engines_stage_2

    dx = np.concatenate((r_dot, v_dot, [m_dot], pr_dot, pv_dot))
    return dx

def simulate_func(optimisation_state,
                  initial_state,
                  time_scale,
                  exo_propelled_lambda,
                  initial_time,
                  return_all_states = False):
    propellant_burn_time = optimisation_state[0] * time_scale
    pr = optimisation_state[1:4]
    pv = optimisation_state[4:7]
    augmented_state = np.concatenate((initial_state, pr, pv))

    t_span = [initial_time, initial_time + propellant_burn_time]
    solution = solve_ivp(exo_propelled_lambda,
                    t_span = t_span,
                    y0 = augmented_state,
                    method='RK23',
                    atol = 1e-3,
                    rtol = 1e-3
                    )
    if not solution.success:
        raise ValueError("ODE integration failed in constraints.")
    final_state = solution.y[0:7, -1]
    if return_all_states:
        states = solution.y
        times = solution.t
        return final_state, states, times
    else:
        return final_state

def initial_opt_state_generation(final_endo_state,
                                 propellant_mass_stage_2_ascent,
                                 mass_flow_stage_2,
                                 time_scale):
    pos = final_endo_state[0:3]
    vel = final_endo_state[3:6]
     
    v_rel = vel - np.cross(w_earth, pos)
    pvU = v_rel / np.linalg.norm(v_rel)

    h_vec = np.cross(pos, vel)                      # Specific angular momentum vector
    omega_orbit = h_vec / (np.linalg.norm(pos)**2)  # Angular velocity of the orbit
    prU = -np.cross(omega_orbit, pvU)

    t_b_total = propellant_mass_stage_2_ascent / mass_flow_stage_2    # Propellant burn time [s]
    t_b_propelled = t_b_total                           # Propellant burn time guess [s]; left over for circularisation
    t_b_scaled = t_b_propelled / time_scale

    initial_optimisation_state = [t_b_scaled,
                                    prU[0],
                                    prU[1],
                                    0,
                                    pvU[0],
                                    pvU[1],
                                    0]
    return initial_optimisation_state

def optimise(initial_state,
             v_exhaust_stage_2,
             mass_flow_per_engine_stage_2,
             number_of_engines_stage_2,
             propellant_mass_stage_2_ascent,
             time_scale,
             mass_burnout,
             semi_major_axis,
             number_of_iterations,
             initial_time,
             mission = 'LEO',
             print_bool = True):
    if mission == 'LEO':
        target_altitude = 200e3
        max_altitude = target_altitude
        minimum_altitude = target_altitude*0.5
    elif mission == 'GEO':
        target_altitude = 35786e3
        max_altitude = target_altitude
        minimum_altitude = target_altitude*0.5
    else:
        raise ValueError("Unknown mission type. Choose 'LEO' or 'GEO'.")
    
    initial_optimisation_state = initial_opt_state_generation(initial_state,
                                                        propellant_mass_stage_2_ascent,
                                                        mass_flow_per_engine_stage_2 * number_of_engines_stage_2,
                                                        time_scale)

    exo_propelled_lambda = lambda t, y: exo_dyn(t, y, v_exhaust_stage_2, mass_flow_per_engine_stage_2, number_of_engines_stage_2)
    final_mass_compute_func_lambda = lambda state: final_mass_compute(state, v_exhaust_stage_2, target_altitude)

    simulate_func_lambda = lambda optimisation_state: simulate_func(optimisation_state,
                                                                    initial_state,
                                                                    time_scale,
                                                                    exo_propelled_lambda,
                                                                    initial_time,
                                                                    return_all_states=True)
    cons = return_constraints(simulate_func_lambda,
                                final_mass_compute_func_lambda,
                                mass_burnout,
                                max_altitude,
                                semi_major_axis,
                                minimum_altitude)
    
    cost_fcn_lambda = lambda optimisation_state: cost_fcn(optimisation_state,
                                                            simulate_func_lambda,
                                                            final_mass_compute_func_lambda,
                                                            mass_burnout,
                                                            semi_major_axis)

    # Create bounds
    optimisation_state_bounds = [
        (initial_optimisation_state[0]*4/5, initial_optimisation_state[0]),        # prop_time_scaled
        (-1, 1),  # pr_x
        (-1, 1),  # pr_y
        (0, 0),  # pr_z
        (-1, 1),                           # pv_x
        (-1, 1),                           # pv_y
        (0, 0)                            # pv_z
    ]

    # Perform optimization
    result = minimize(
        fun=cost_fcn_lambda,
        x0=initial_optimisation_state ,
        bounds=optimisation_state_bounds,
        constraints=cons,
        method='trust-constr',
        options={
            'disp': True,
            'maxiter': number_of_iterations,
            'initial_tr_radius': 10,
            'gtol': 1e-2,
            'xtol': 1e-2
        }
    )
    optimised_state = result.x
    propellant_burn_time = optimised_state[0] * time_scale  # Optimized propellant time in seconds
    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization did not converge:", result.message)
    
    prU = optimised_state[1:4]
    pvU = optimised_state[4:7]
    if print_bool:
        print("Optimization successful!")
        print(f"Optimized propellant time: {propellant_burn_time} s")
        print(f"Optimized pr vector: {prU}")
        print(f"Optimized pv vector: {pvU}")
    
    final_state, states, times = simulate_func(optimised_state,
                                                initial_state,
                                                time_scale,
                                                exo_propelled_lambda,
                                                initial_time,
                                                return_all_states=True)


    optimised_state[0] = propellant_burn_time * time_scale  # Convert back to optimised propellant time

    return optimised_state, final_state, states, times



def exp_opt_lambda_func_creation(config,
                                 stage_masses_dict,
                                 print_bool = True):
    v_exhaust_stage_2 = config['v_exhaust'][1]
    thrust_per_engine_stage_2 = config['engine_dictionaries'][1]['thrust_N']
    mass_flow_per_engine_stage_2 = thrust_per_engine_stage_2 / v_exhaust_stage_2
    semi_major_axis = config['target_altitude'] + R_earth
    mission = config['mission']
    time_scale = 100
    number_of_iterations = 1000

    initial_optimisation_state_lambda_func = lambda final_state_endo : \
        initial_opt_state_generation(final_state_endo,
                                    stage_masses_dict['propellant_mass_stage_2_ascent'],
                                    mass_flow_per_engine_stage_2,
                                    time_scale)

    exo_opt_lambda_func = lambda final_state_endo, number_of_engines_stage_2, initial_time : \
        optimise(final_state_endo,
                 initial_optimisation_state_lambda_func(final_state_endo),
                 v_exhaust_stage_2,
                 mass_flow_per_engine_stage_2,
                 number_of_engines_stage_2,
                 time_scale,
                 stage_masses_dict['mass_at_stage_2_ascent_burnout'],
                 semi_major_axis,
                 number_of_iterations,
                 initial_time,
                 mission,
                 print_bool)
    
    return exo_opt_lambda_func

def post_process_exo_propelled_opt(augmented_states_exo,
                                   times_exo,
                                   previous_times,
                                   previous_states):
    # Transpose previous_states to ensure it has the correct shape
    times = previous_times

    states_exo = np.zeros((7, len(times_exo)))  # Transpose shape
    altitudes_exo = np.zeros(len(times_exo))
    speeds_exo = np.zeros(len(times_exo))
    masses_exo = np.zeros(len(times_exo))
    R_earth = 6371e3
    
    for i, t in enumerate(times_exo):
        state_exo_pure = augmented_states_exo[:, i][0:7] # Un-augmented state
        vel_exo = np.linalg.norm(state_exo_pure[3:6])
        alt_exo = np.linalg.norm(state_exo_pure[0:3]) - R_earth
        mass_exo = state_exo_pure[6]
        altitudes_exo[i] = alt_exo
        speeds_exo[i] = vel_exo
        masses_exo[i] = mass_exo
        states_exo[:, i] = state_exo_pure  # Transpose shape
        if i != 0: # Ignore initial condition
            times = np.append(times, times_exo[i])
    states = np.concatenate((previous_states, states_exo), axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(times_exo, altitudes_exo/1000)
    axs[0].set_ylabel('Altitude [km]')
    axs[0].set_xlabel('Time [s]')
    axs[1].plot(times_exo, speeds_exo/1000)
    axs[1].set_ylabel('Velocity [km/s]')
    axs[1].set_xlabel('Time [s]')
    axs[2].plot(times_exo, masses_exo)
    axs[2].set_ylabel('Mass [kg]')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('results/exo_propelled_opt.png')
    plt.close()

    return states_exo, times_exo, states, times

def exo_propelled(initial_state,
                  previous_times,
                  previous_states,
                  v_exhaust_stage_2,
                  mass_flow_per_engine_stage_2,
                  number_of_engines_stage_2,
                  propellant_mass_stage_2_ascent,
                  time_scale,
                  mass_burnout,
                  semi_major_axis,
                  number_of_iterations,
                  mission = 'LEO',
                  print_bool = True):
    
    optimised_state, final_state, states, times = optimise(initial_state,
             v_exhaust_stage_2,
             mass_flow_per_engine_stage_2,
             number_of_engines_stage_2,
             propellant_mass_stage_2_ascent,
             time_scale,
             mass_burnout,
             semi_major_axis,
             number_of_iterations,
             previous_times[-1],
             mission,
             print_bool)
    
    states_exo, times_exo, states, times = post_process_exo_propelled_opt(states,
                                                                          times,
                                                                          previous_times,
                                                                          previous_states)
    
    return states_exo, times_exo, states, times, final_state
