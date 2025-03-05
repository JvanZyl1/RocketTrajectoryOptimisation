import numpy as np
import scipy
from TrajectoryGeneration.atmosphere import endo_atmospheric_model
import matplotlib.pyplot as plt

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def rocket_dynamics(t,
                    state_vector,
                    mass_flow_endo,                     # All engines combined mass flow rate [kg/s]
                    specific_impulse_vacuum,
                    get_drag_coefficient_func,
                    frontal_area,
                    nozzle_exit_area,
                    nozzle_exit_pressure,
                    number_of_engines):
    pos = state_vector[:3]
    vel = state_vector[3:6]
    m = state_vector[6]
    alt = np.linalg.norm(pos) - R_earth
    rho, p_atm, a = endo_atmospheric_model(alt)
    vel_rel = vel - np.cross(w_earth, pos)
    mach = np.linalg.norm(vel_rel) / a
    cd = get_drag_coefficient_func(mach)
    thrust = specific_impulse_vacuum * g0 * mass_flow_endo + \
             (nozzle_exit_pressure - p_atm) * nozzle_exit_area * number_of_engines
    drag = 0.5 * rho * (np.linalg.norm(vel_rel)**2) * frontal_area * cd
    r_dot = vel
    v_dot = (-mu / (np.linalg.norm(pos)**3)) * pos \
            + (thrust / m) * (pos / np.linalg.norm(pos)) \
            - (drag / m) * (pos / np.linalg.norm(pos))
    dm = -mass_flow_endo
    return np.concatenate((r_dot, v_dot, [dm]))

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

def negative_altitude_event():
    def negative_altitude_event(t, y):
        altitude = np.linalg.norm(y[:3]) - R_earth
        return altitude + 1  # Will trigger when altitude < -1m 
    negative_altitude_event.terminal = True
    return negative_altitude_event

def make_events(target_altitude, minimum_mass):
    return [make_altitude_event(target_altitude), make_mass_flow_event(minimum_mass), negative_altitude_event()]


def vertical_rising_initial_state(initial_mass):
    R_earth = 6378137  # Earth radius [m]
    w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]                            
    position_vector_initial = np.array([R_earth, 0, 0])       # Initial position vector [m]
    velocity_vector_initial = np.cross(w_earth, position_vector_initial)                                  # Initial velocity vector [m/s]

    initial_state_vertical_rising = [
        position_vector_initial[0],
        position_vector_initial[1],
        position_vector_initial[2],
        velocity_vector_initial[0],
        velocity_vector_initial[1],
        velocity_vector_initial[2],
        initial_mass
    ]

    # Unit east vector
    unit_position_vector_initial = position_vector_initial / np.linalg.norm(position_vector_initial)      # Initial position unit vector
    east_vector = np.cross([0, 0, 1], unit_position_vector_initial)                                       # East vector [m]
    unit_east_vector = east_vector / np.linalg.norm(east_vector)                                          # East unit vector
    return initial_state_vertical_rising, unit_east_vector


def endo_atmospheric_vertical_rising(initial_mass,
                                    target_altitude,
                                    minimum_mass,
                                    mass_flow_endo,
                                    specfic_impulse_vacuum,
                                    get_drag_coefficient_func,
                                    frontal_area,
                                    nozzle_exit_area,
                                    nozzle_exit_pressure,
                                    number_of_engines):
    
    initial_state, unit_east_vector = vertical_rising_initial_state(initial_mass)
    rocket_dynamics_lambda = lambda t, y: rocket_dynamics(t,
                                                          y,
                                                          mass_flow_endo,
                                                          specfic_impulse_vacuum,
                                                          get_drag_coefficient_func,
                                                          frontal_area,
                                                          nozzle_exit_area,
                                                          nozzle_exit_pressure,
                                                          number_of_engines)

    # Mock t_span to cover all events
    t_span = [0, 10000]
    sol = scipy.integrate.solve_ivp (
        rocket_dynamics_lambda,
        t_span,  
        initial_state,
        events=make_events(target_altitude, minimum_mass), 
        max_step=0.1,
        rtol=1e-8,
        atol=1e-8
    )

    final_state = sol.y[:, -1]

    # Calculate relative velocity correctly by handling the cross product for each time step
    velocities = sol.y[3:6]
    positions = sol.y[:3]
    relative_velocities = np.zeros(len(sol.t))
    
    for i in range(len(sol.t)):
        vel = velocities[:, i]
        pos = positions[:, i]
        relative_velocities[i] = np.linalg.norm(vel - np.cross(w_earth, pos))

    altitude = np.linalg.norm(sol.y[:3], axis=0) - R_earth
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(sol.t, altitude)
    axs[0].set_ylabel('Altitude [m]')
    axs[0].set_xlabel('Time [s]')
    axs[1].plot(sol.t, relative_velocities)
    axs[1].set_ylabel('Relative Velocity [m/s]')
    axs[1].set_xlabel('Time [s]')
    axs[2].plot(sol.t, sol.y[6])
    axs[2].set_ylabel('Mass [kg]')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('results/vertical_rising.png')
    plt.close()

    return sol.t, sol.y, final_state, unit_east_vector