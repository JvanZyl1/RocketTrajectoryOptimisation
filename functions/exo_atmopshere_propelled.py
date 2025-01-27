import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from functions.params import mu, g0, R_earth, specific_impulses_vacuum

def exo_dyn(t,
            augmented_state_vector,
            mass_flow_exo):
    """
    Defines the ODE system.
    x = [r_x, r_y, r_z, v_x, v_y, v_z, m, p_r_x, p_r_y, p_r_z, p_v_x, p_v_y, p_v_z]
    """
    thrust = specific_impulses_vacuum[1] * g0 * mass_flow_exo 
    r = augmented_state_vector[0:3]
    v = augmented_state_vector[3:6]
    m = augmented_state_vector[6]
    p_r = augmented_state_vector[7:10]
    p_v = augmented_state_vector[10:13]


    r_dot = v
    v_dot = (-mu / np.linalg.norm(r)**3) * r + (thrust / m) * (p_v / np.linalg.norm(p_v))
    pr_dot = - (mu / np.linalg.norm(r)**3) * (3 * np.dot(p_v, r) * r / np.linalg.norm(r)**2 - p_v)
    pv_dot = -p_r

    m_dot = -mass_flow_exo

    dx = np.concatenate((r_dot, v_dot, [m_dot], pr_dot, pv_dot))
    return dx

def exo_atmosphere_propelled(initial_state,
                             optimisation_parameters,
                             t_start,
                             mass_flow_exo,
                             save_file_path = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results',
                             plot_bool=False):
    """
    Propagate the state vector from the exo-atmosphere to the orbit.
    """
    # Unpack the initial state
    r0 = initial_state[0:3]
    v0 = initial_state[3:6]
    m0 = initial_state[6]

    # Unpack the optimisation parameters
    t_burn = optimisation_parameters[0]
    prU = optimisation_parameters[1:4]
    pvU = optimisation_parameters[4:7]

    # Define the initial augmented state vector
    augmented_state_vector = np.concatenate((r0, v0, [m0], prU, pvU))

    # Define the time span
    t_span = [t_start, t_start + t_burn]
    


    # Propagate the state vector
    sol = solve_ivp(lambda t, y: exo_dyn(t, y, mass_flow_exo),
                    t_span=t_span,
                    y0=augmented_state_vector,
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-10)
    
    # Extract the final state vector
    final_state = sol.y[:, -1]

    # Extract the states
    states = sol.y

    # Extract the times
    times = sol.t

    # Extract variables
    m = states[6, :]

    altitude =  np.linalg.norm(states[0:3, :], axis=0) - R_earth
    speed = np.linalg.norm(states[3:6, :], axis=0)
    # Plot the altitude, make thin but long plot as in high not wide
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(times, altitude/1000)
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [km]')
    plt.grid()
    plt.subplot(1,3,2)
    plt.plot(times, speed/1000)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [km/s]')
    plt.grid()
    plt.subplot(1,3,3)
    plt.plot(times, m)
    plt.xlabel('Time [s]')
    plt.ylabel('m [kg]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_file_path + '/exo_atmosphere_propelled_states.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()


    return final_state, states, times