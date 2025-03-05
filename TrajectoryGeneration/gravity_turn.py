import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from TrajectoryGeneration.vertical_rising import make_events
from TrajectoryGeneration.atmosphere import endo_atmospheric_model

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def rocket_dynamics(t,
                    state_vector,
                    mass_flow_endo,                     # All engines mass flow rate [kg/s]
                    specific_impulse_vacuum,
                    get_drag_coefficient_func,
                    frontal_area,
                    nozzle_exit_area,
                    nozzle_exit_pressure,
                    number_of_engines,
                    simulation_bool = False):
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

    vel_rel_norm = np.linalg.norm(vel_rel)
    vel_rel_unit_vec = vel_rel / vel_rel_norm
    

    gravity = -mu / (np.linalg.norm(pos)**3) * pos

    r_dot = vel
    v_dot = gravity \
            + (thrust / m) * vel_rel_unit_vec \
            - (drag / m) * vel_rel_unit_vec
    dm = -mass_flow_endo

    return np.concatenate((r_dot, v_dot, [dm]))


def gravity_turn_initial_state(vertical_rising_final_state,
                               kick_angle,
                               unit_east_vector):

    pos = vertical_rising_final_state[:3]
    vel = vertical_rising_final_state[3:6]
    m = vertical_rising_final_state[6]

    vel_x = vel[0]
    vel_y = vel[1]
    vel_z = vel[2]

    vel_x_new = vel_x * np.cos(kick_angle) - vel_y * np.sin(kick_angle)
    vel_y_new = vel_x * np.sin(kick_angle) + vel_y * np.cos(kick_angle)
    vel_z_new = vel_z

    vel_0 = np.array([vel_x_new, vel_y_new, vel_z_new])

    return np.concatenate((pos, vel_0, [m]))

# Mock t_span to cover all events
def endo_atmospheric_gravity_turn(vertical_rising_final_state,
                                  kick_angle,
                                  unit_east_vector,
                                  t_start: float,
                                  target_altitude: float,
                                  minimum_mass: float,
                                  mass_flow_endo: float,
                                  specific_impulse_vacuum: float,
                                  get_drag_coefficient_func,
                                  frontal_area: float,
                                  nozzle_exit_area: float,
                                  nozzle_exit_pressure: float,
                                  number_of_engines: int,
                                  thrust_throttle: float,
                                  thrust_altitudes: tuple):
    initial_state = gravity_turn_initial_state(vertical_rising_final_state,
                                              kick_angle,
                                              unit_east_vector)


    rocket_dynamics_un_throttled_lambda = lambda t, y: rocket_dynamics(t,
                                                          y,
                                                          mass_flow_endo=mass_flow_endo,
                                                          specific_impulse_vacuum=specific_impulse_vacuum,
                                                          get_drag_coefficient_func=get_drag_coefficient_func,
                                                          frontal_area=frontal_area,
                                                          nozzle_exit_area=nozzle_exit_area,
                                                          nozzle_exit_pressure=nozzle_exit_pressure,
                                                          number_of_engines=number_of_engines)
    mass_flow_throttled = mass_flow_endo * thrust_throttle
    rocket_dynamics_throttled_lambda = lambda t, y: rocket_dynamics(t,
                                                          y,
                                                          mass_flow_endo=mass_flow_throttled,
                                                          specific_impulse_vacuum=specific_impulse_vacuum,
                                                          get_drag_coefficient_func=get_drag_coefficient_func,
                                                          frontal_area=frontal_area,
                                                          nozzle_exit_area=nozzle_exit_area,
                                                          nozzle_exit_pressure=nozzle_exit_pressure,
                                                          number_of_engines=number_of_engines)
    t_span_unthrottled_1 = [t_start, 10000]
    
    # First solve unthrottled dynamics to thrust_altitudes[0]
    sol_unthrottled_1 = solve_ivp(fun=rocket_dynamics_un_throttled_lambda,
                                  t_span=t_span_unthrottled_1,  
                                  y0=initial_state,
                                  events=make_events(thrust_altitudes[0], minimum_mass), 
                                  max_step=0.1,  # limiting step size for demonstration
                                  rtol=1e-8,
                                  atol=1e-8)

    final_state_unthrottled_1 = sol_unthrottled_1.y[:, -1]
    final_time_unthrottled_1 = sol_unthrottled_1.t[-1]

    states = sol_unthrottled_1.y
    times = sol_unthrottled_1.t

    # Second solve throttled dynamics from thrust_altitudes[0] to thrust_altitudes[1]
    t_span_throttled = [final_time_unthrottled_1, 10000]
    sol_throttled_1 = solve_ivp(fun=rocket_dynamics_throttled_lambda,
                               t_span=t_span_throttled,
                               y0=final_state_unthrottled_1,
                               events=make_events(thrust_altitudes[1], minimum_mass),
                               max_step=0.1,
                               rtol=1e-8,
                               atol=1e-8)
    
    final_state_throttled_1 = sol_throttled_1.y[:, -1]
    final_time_throttled_1 = sol_throttled_1.t[-1]

    states = np.concatenate((states, sol_throttled_1.y), axis=1)
    times = np.concatenate((times, sol_throttled_1.t))

    # Third solve unthrottled dynamics from thrust_altitudes[1] to target_altitude
    t_span_unthrottled_2 = [final_time_throttled_1, 10000]
    sol_unthrottled_2 = solve_ivp(fun=rocket_dynamics_un_throttled_lambda,
                                  t_span=t_span_unthrottled_2,
                                  y0=final_state_throttled_1,
                                  events=make_events(target_altitude, minimum_mass),
                                  max_step=0.1,  # limiting step size for demonstration
                                  rtol=1e-8,
                                  atol=1e-8)
    
    final_state = sol_unthrottled_2.y[:, -1]

    states = np.concatenate((states, sol_unthrottled_2.y), axis=1)
    times = np.concatenate((times, sol_unthrottled_2.t))


    # Plot atitude, velocity, and fuel mass next to each other
    velocities = states[3:6]
    positions = states[:3]
    relative_velocities = np.zeros(len(times))
    
    for i in range(len(times)):
        vel = velocities[:, i]
        pos = positions[:, i]
        relative_velocities[i] = np.linalg.norm(vel - np.cross(w_earth, pos))

    altitude = np.linalg.norm(states[:3], axis=0) - R_earth
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(times, altitude)
    axs[0].set_ylabel('Altitude [km]')
    axs[0].set_xlabel('Time [s]')
    axs[1].plot(times, relative_velocities/1000)
    axs[1].set_ylabel('Relative Velocity [km/s]')
    axs[1].set_xlabel('Time [s]')
    axs[2].plot(times, states[6])
    axs[2].axhline(y=minimum_mass, color='r', linestyle='--')
    axs[2].set_ylabel('Mass [kg]')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('results/gravity_turn.png')
    plt.close()
    # Now  plot the dynamic pressure
    dynamic_pressures = np.zeros(len(times))
    for i, time in enumerate(times):
        state = states[:, i]
        pos = state[:3]
        vel = state[3:6]
        alt = np.linalg.norm(pos) - R_earth
        rho, p_atm, a = endo_atmospheric_model(alt)
        vel_rel = vel - np.cross(w_earth, pos)
        dynamic_pressures[i] = 0.5 * rho * np.linalg.norm(vel_rel)**2

    plt.figure()
    plt.plot(times, dynamic_pressures)
    plt.xlabel('Time [s]')
    plt.ylabel('Dynamic Pressure [Pa]')
    plt.title('Dynamic Pressure during gravity turn')
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/dynamic_pressure.png')
    plt.close()

    max_dynamic_pressure = np.max(dynamic_pressures)

    return times, states, final_state, max_dynamic_pressure

