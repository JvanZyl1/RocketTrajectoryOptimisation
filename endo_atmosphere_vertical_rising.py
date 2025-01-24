import numpy as np
from scipy.integrate import solve_ivp
from params import (rho0, scale_height_endo, R_earth, w_earth, g0, mu,
                    aerodynamic_area, nozzle_exit_area, nozzle_exit_pressure,
                    get_drag_coefficient, specific_impulses_vacuum)
import matplotlib.pyplot as plt

def endo_atmospheric_model(h):
    if h >= 0:
        rho = rho0 * np.exp(-h / scale_height_endo)
        P_a = 101325 * (rho / rho0)
    else:
        rho = rho0
        P_a = 101325
    a = 340.29
    return rho, P_a, a

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
            + (thrust / m) * (pos / np.linalg.norm(pos)) \
            - (drag / m) * (pos / np.linalg.norm(pos))
    dm = -mass_flow_endo
    return np.concatenate((r_dot, v_dot, [dm]))

'''
fROM DOCUMENTATION
eventscallable, or list of callables, optional
Events to track. If None (default), no events will be tracked.
Each event occurs at the zeros of a continuous function of time and state.
Each function must have the signature event(t, y) where additional argument have to be passed if args is used (see documentation of args argument).
Each function must return a float. The solver will find an accurate value of t at which event(t, y(t)) = 0 using a root-finding algorithm.
By default, all zeros will be found. The solver looks for a sign change over each step, so if multiple zero crossings occur within one step,
events may be missed. Additionally each event function might have the following attributes:

terminal: bool or int, optional
When boolean, whether to terminate integration if this event occurs.
When integral, termination occurs after the specified the number of occurrences of this event.
Implicitly False if not assigned.
'''
# Make altitude and fuel mass events
def make_altitude_event(target_altitude):
    def altitude_event(t, y):
        altitude = np.linalg.norm(y[:3]) - R_earth
        return altitude - target_altitude
    altitude_event.terminal = True
    return altitude_event

def make_mass_flow_event(minimum_mass):
    def mass_flow_event(t, y):
        return y[6] - minimum_mass
    mass_flow_event.terminal = True
    return mass_flow_event

def make_events(target_altitude, minimum_mass):
    return [make_altitude_event(target_altitude), make_mass_flow_event(minimum_mass)]

def endo_atmospheric_vertical_rising(initial_state,
                                    target_altitude,
                                    minimum_mass,
                                    mass_flow_endo,
                                    plot_bool = False,
                                    save_file_path = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
                            
    # Mock t_span to cover all events
    t_span = [0, 10000]
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
    plt.savefig(save_file_path + '/endo_atmospheric_vertical_rising.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    return sol.t, sol.y, final_state