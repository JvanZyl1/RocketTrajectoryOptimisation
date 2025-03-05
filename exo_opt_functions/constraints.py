
import numpy as np

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]


def mass_constraint_fcn(optimisation_state,
                        simulate_func_lambda,
                        final_mass_compute_func,
                        dry_mass):
    
    final_state = simulate_func_lambda(optimisation_state)
    
    mass_final_at_circular = final_mass_compute_func(final_state)
    mass_constraint = mass_final_at_circular - dry_mass # >= 0

    return mass_constraint

def altitude_constraint_fcn(optimisation_state,
                            simulate_func_lambda,
                            max_altitude):
    final_state = simulate_func_lambda(optimisation_state)
    
    final_position_exo_arrival = final_state[0:3]
    final_altitude_exo_arrival = np.linalg.norm(final_position_exo_arrival) - R_earth

    altitude_constraint = max_altitude - final_altitude_exo_arrival  # >= 0

    return altitude_constraint

def final_altitude_constraint(optimisation_state,
                              simulate_func_lambda,
                              minimum_altitude):
    final_state = simulate_func_lambda(optimisation_state)

    final_position_exo_arrival = final_state[0:3]
    final_altitude_exo_arrival = np.linalg.norm(final_position_exo_arrival) - R_earth

    # Temporary changining
    minimum_altitude_constraint = final_altitude_exo_arrival - minimum_altitude # >= 0

    return minimum_altitude_constraint

def return_constraints(simulate_func_lambda,
                       final_mass_compute_func_lambda,
                       dry_mass,
                       max_altitude,
                       semi_major_axis,
                       minimum_altitude):
    mass_constraint_lambda_func = lambda optimisation_state: mass_constraint_fcn(optimisation_state,
                                                                                 simulate_func_lambda,
                                                                                 final_mass_compute_func_lambda,
                                                                                 dry_mass)

    altitude_constraint_lambda_func = lambda optimisation_state: altitude_constraint_fcn(optimisation_state,
                                                                                         simulate_func_lambda,
                                                                                         max_altitude)

    
    final_altitude_constraint_lambda_func = lambda optimisation_state: final_altitude_constraint(optimisation_state,
                                                                                                 simulate_func_lambda,
                                                                                                 minimum_altitude)

    cons = [{'type':'ineq', 'fun': mass_constraint_lambda_func},
            {'type':'ineq', 'fun': altitude_constraint_lambda_func},
            {'type':'ineq', 'fun': final_altitude_constraint_lambda_func}]
    
    return cons