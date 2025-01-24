import numpy as np
from params import mu, R_earth
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coasting_derivatives(t, y):
    r = y[:3]
    v = y[3:6]
    m = y[6]
    rdot = v
    vdot = -mu / (np.linalg.norm(r) ** 3) * r
    return np.concatenate((rdot, vdot, [0]))

def endo_atmosphere_coasting(t_start: float,
                             initial_state: np.ndarray,
                             time_stopping: float,
                             plot_bool: bool = False,
                             save_file_path: str = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
    t_span = [t_start, time_stopping]
    sol = solve_ivp(
        coasting_derivatives,  # Use the renamed ODE function
        t_span,
        initial_state,
        max_step=0.1,
        rtol=1e-8,
        atol=1e-8
    )

    final_state = sol.y[:, -1]

    if plot_bool:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(sol.t, (np.linalg.norm(sol.y[:3], axis=0) - R_earth)/1000)
        axs[0].set_ylabel('Altitude [km]')
        axs[0].set_xlabel('Time [s]')
        axs[1].plot(sol.t, (np.linalg.norm(sol.y[3:6], axis=0))/1000)
        axs[1].set_ylabel('Velocity [km/s]')
        axs[1].set_xlabel('Time [s]')
        axs[2].plot(sol.t, sol.y[6])
        axs[2].set_ylabel('Mass [kg]')
        axs[2].set_xlabel('Time [s]')
        plt.show()

    return sol.t, sol.y, final_state
