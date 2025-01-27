import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.params import mu, R_earth


def coasting_derivatives(t, y):
    r = y[:3]
    v = y[3:6]
    m = y[6]
    rdot = v
    vdot = -mu / (np.linalg.norm(r) ** 3) * r
    return np.concatenate((rdot, vdot, [0]))

def make_altitude_event(target_altitude):
    def altitude_event(t, y):
        altitude = np.linalg.norm(y[:3]) - R_earth
        return altitude - target_altitude
    altitude_event.terminal = True
    return altitude_event

def exo_atmosphere_coasting_to_orbit(t_start,
                                     initial_state, # [r, v, m]
                                     target_altitude,
                                     plot_bool = False,
                                     save_file_path = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
    
    # Mock t_span to cover all events
    t_span = [t_start, 10000]
    sol = solve_ivp(
        coasting_derivatives,
        t_span,  
        initial_state,
        events=make_altitude_event(target_altitude), 
        max_step=0.1,  # limiting step size for demonstration
        rtol=1e-8,
        atol=1e-8
    )

    final_state = sol.y[:, -1]

    # Plot atitude, velocity, and fuel mass next to each other

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(sol.t, np.linalg.norm(sol.y[:3], axis=0) - R_earth)
    axs[0].set_ylabel('Altitude [m]')
    axs[0].set_xlabel('Time [s]')
    axs[1].plot(sol.t, np.linalg.norm(sol.y[3:6], axis=0))
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_xlabel('Time [s]')
    axs[2].plot(sol.t, sol.y[6])
    axs[2].set_ylabel('Mass [kg]')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(save_file_path + '/exo_atmophere_coasting_to_orbit.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    return sol.t, sol.y, final_state