
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from TrajectoryGeneration.atmosphere import endo_atmospheric_model

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def coasting_derivatives(t,
                         y,
                         get_drag_coefficient_func,
                         frontal_area):
    r = y[:3]
    v = y[3:6]
    m = y[6]
    alt = np.linalg.norm(r) - R_earth
    rho, p_atm, a = endo_atmospheric_model(alt)
    vel_rel = v - np.cross(w_earth, r)

    mach = np.linalg.norm(vel_rel) / a
    cd = get_drag_coefficient_func(mach)
    drag = 0.5 * rho * (np.linalg.norm(vel_rel)**2) * frontal_area * cd

    rdot = v
    vdot = -mu / (np.linalg.norm(r) ** 3) * r \
        - (drag / m) * (vel_rel / np.linalg.norm(vel_rel))
    return np.concatenate((rdot, vdot, [0]))

def endo_coasting_sub_func(t_start: float,
                           initial_state: np.ndarray,
                           time_stopping: float,
                           frontal_area: float,
                           get_drag_coefficient_func: callable):

    t_span = [t_start, time_stopping]
    coasting_lambda_func = lambda t, y : coasting_derivatives(t,
                                                              y,
                                                              get_drag_coefficient_func,
                                                              frontal_area)
    sol = solve_ivp(
        coasting_lambda_func,
        t_span,
        initial_state,
        max_step=0.1,
        rtol=1e-8,
        atol=1e-8
    )

    final_state = sol.y[:, -1]


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
    plt.tight_layout()
    plt.savefig('results/endo_coasting.png')
    plt.close()

    return sol.t, sol.y, final_state


def endo_coasting(previous_times,
                  previous_states,
                  rocket_mass, 
                  coasting_time,
                  frontal_area,
                  get_drag_coefficient_func):
    
    # Coasting initial conditions
    start_time = previous_times[-1]
    start_state_full_rocket = previous_states[:, -1]

    # This includes payload fairing
    mass_of_stage_2 = rocket_mass
    start_state_stage_2 = start_state_full_rocket
    start_state_stage_2[-1] = mass_of_stage_2

    # Coasting
    end_time = start_time + coasting_time
    coasting_times, coasting_states, final_state = endo_coasting_sub_func(start_time,
                                                                          start_state_stage_2,
                                                                          end_time,
                                                                          frontal_area,
                                                                          get_drag_coefficient_func)
    
    # Update times and states np arrays
    states = np.concatenate((previous_states, coasting_states), axis=1)
    times = np.concatenate((previous_times, coasting_times))

    return times, states, final_state, coasting_times, coasting_states

