import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Make altitude and fuel mass events
from endo_atmosphere_vertical_rising import make_events, endo_atmospheric_model
from params import (R_earth, w_earth, g0, mu, nozzle_exit_pressure, \
                     nozzle_exit_area, aerodynamic_area, specific_impulses_vacuum, \
                     get_drag_coefficient)

def rocket_dynamics(t, state_vector, mass_flow_endo):
    pos = state_vector[:3]
    vel = state_vector[3:6]
    m = state_vector[6]
    alt = np.linalg.norm(pos) - R_earth
    rho, p_atm, a = endo_atmospheric_model(alt)
    vel_rel = vel - np.cross(w_earth, pos)
    mach = np.linalg.norm(vel_rel) / a
    cd = get_drag_coefficient(mach)
    thrust = specific_impulses_vacuum[0] * g0 * mass_flow_endo + \
             (nozzle_exit_pressure - p_atm) * nozzle_exit_area
    drag = 0.5 * rho * (np.linalg.norm(vel_rel)**2) * aerodynamic_area * cd
    r_dot = vel
    v_dot = (-mu / (np.linalg.norm(pos)**3)) * pos \
            + (thrust / m) * (vel_rel / np.linalg.norm(vel_rel)) \
            - (drag / m) * (vel_rel / np.linalg.norm(vel_rel))
    dm = -mass_flow_endo
    return np.concatenate((r_dot, v_dot, [dm]))

# Mock t_span to cover all events
def endo_atmospheric_gravity_turn(t_start: float,
                                    initial_state: np.ndarray,
                                    target_altitude: float,
                                    minimum_mass: float,
                                    mass_flow_endo: float,
                                    plot_bool: bool = False,
                                    save_file_path: str = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
    t_span = [t_start, 10000]
    # solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
    sol = solve_ivp(
        lambda t, y: rocket_dynamics(t, y, mass_flow_endo),
        t_span,  
        initial_state,
        events=make_events(target_altitude, minimum_mass), 
        max_step=0.1,  # limiting step size for demonstration
        rtol=1e-8,
        atol=1e-8
    )

    final_state = sol.y[:, -1]

    # Plot atitude, velocity, and fuel mass next to each other

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
    plt.savefig(save_file_path + '/endo_atmospheric_gravity_turn.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    return sol.t, sol.y, final_state