import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from params import mu, g0, R_earth

def exo_dyn(t,
            augmented_state_vector,
            Isp,
            mass_flow_exo):
    """
    Defines the ODE system.
    x = [r_x, r_y, r_z, v_x, v_y, v_z, m, p_r_x, p_r_y, p_r_z, p_v_x, p_v_y, p_v_z]
    """
    thrust = Isp * g0 * mass_flow_exo
    r = augmented_state_vector[0:3]
    v = augmented_state_vector[3:6]
    m = augmented_state_vector[6]
    p_r = augmented_state_vector[7:10]
    p_v = augmented_state_vector[10:13]

    r_norm = np.linalg.norm(r)
    if np.linalg.norm(p_v) == 0:
        p_v_norm = 1e-8
    else:
        p_v_norm = np.linalg.norm(p_v)

    r_dot = v
    v_dot = (-mu / r_norm**3) * r + (thrust / m) * (p_v / p_v_norm)
    pr_dot = - (mu / r_norm**3) * (3 * np.dot(p_v, r) * r / r_norm**2 - p_v)
    pv_dot = -p_r

    m_dot = -mass_flow_exo

    dx = np.concatenate((r_dot, v_dot, [m_dot], pr_dot, pv_dot))
    return dx

def exo_atmosphere_propelled(initial_state,
                             optimisation_parameters,
                             t_start,
                             Isp,
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
    sol = solve_ivp(lambda t, y: exo_dyn(t, y, Isp, mass_flow_exo),
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
    r_x = states[0, :]
    r_y = states[1, :]
    r_z = states[2, :]
    v_x = states[3, :]
    v_y = states[4, :]
    v_z = states[5, :]
    m = states[6, :]
    p_r_x = states[7, :]
    p_r_y = states[8, :]
    p_r_z = states[9, :]
    p_v_x = states[10, :]
    p_v_y = states[11, :]
    p_v_z = states[12, :]


    altitude =  np.linalg.norm(states[0:3, :], axis=0) - R_earth
    # Plot the altitude
    plt.figure()
    plt.plot(times, altitude)
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [m]')
    plt.grid()
    plt.savefig(save_file_path + '/exo_atmosphere_propelled_altitude.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    # Plot the results [r, v, m, p_r, p_v]
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, r_x)
    plt.xlabel('Time [s]')
    plt.ylabel('r_x [m]')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(times, r_y)
    plt.xlabel('Time [s]')
    plt.ylabel('r_y [m]')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(times, r_z)
    plt.xlabel('Time [s]')
    plt.ylabel('r_z [m]')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(times, v_x)
    plt.xlabel('Time [s]')
    plt.ylabel('v_x [m/s]')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(times, v_y)
    plt.xlabel('Time [s]')
    plt.ylabel('v_y [m/s]')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(times, v_z)
    plt.xlabel('Time [s]')
    plt.ylabel('v_z [m/s]')
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_file_path + '/exo_atmosphere_propelled_states.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.plot(times, m)
    plt.xlabel('Time [s]')
    plt.ylabel('m [kg]')
    plt.grid()
    plt.savefig(save_file_path + '/exo_atmosphere_propelled_mass.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, p_r_x)
    plt.xlabel('Time [s]')
    plt.ylabel('p_r_x [m]')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(times, p_r_y)
    plt.xlabel('Time [s]')
    plt.ylabel('p_r_y [m]')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(times, p_r_z)
    plt.xlabel('Time [s]')
    plt.ylabel('p_r_z [m]')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(times, p_v_x)
    plt.xlabel('Time [s]')
    plt.ylabel('p_v_x [m/s]')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(times, p_v_y)
    plt.xlabel('Time [s]')
    plt.ylabel('p_v_y [m/s]')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(times, p_v_z)
    plt.xlabel('Time [s]')
    plt.ylabel('p_v_z [m/s]')
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_file_path + '/exo_atmosphere_propelled_adjoints.png')
    if plot_bool:
        plt.show()
    else:
        plt.close


    return final_state, states, times