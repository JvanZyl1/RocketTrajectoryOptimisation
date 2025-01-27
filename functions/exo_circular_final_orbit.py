import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from functions.params import G, M_earth, R_earth, specific_impulses_vacuum, mu, g0

# Define Two-Body Equations of Motion
def coasting_derivatives(t, y):
    r = y[:3]
    v = y[3:6]
    m = y[6]
    rdot = v
    vdot = -mu / (np.linalg.norm(r) ** 3) * r
    return np.concatenate((rdot, vdot, [0]))

def final_orbit_maneuver(initial_state,
                         altitude_orbit,
                         t_start,
                         plot_bool=False,
                         save_file_path='/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
    initial_position_vector = initial_state[:3]
    initial_velocity_vector = initial_state[3:6]
    initial_mass = initial_state[6]

    initial_speed = np.linalg.norm(initial_velocity_vector)  # Current speed [m/s]
    orbit_velocity = np.sqrt(G * M_earth / (R_earth + altitude_orbit))  # Required orbital velocity [m/s]
    delta_v_required = orbit_velocity - initial_speed  # Delta V required [m/s]
    Isp_final_stage = specific_impulses_vacuum[1]  # Specific impulse [s]

    # Calculate propellant mass needed
    delta_f = delta_v_required / (Isp_final_stage * g0) # Propellant mass fraction
    final_mass = initial_mass * np.exp(-delta_f)
    change_in_mass = initial_mass - final_mass

    # Apply Delta V in the prograde direction
    unit_velocity = initial_velocity_vector / initial_speed
    delta_v_vector = delta_v_required * unit_velocity
    final_speed = initial_velocity_vector + delta_v_vector

    # Update final state
    final_state = np.concatenate((initial_position_vector, final_speed, [final_mass]))


    #print(f"Delta V Required: {delta_v_required:.2f} m/s")
    #print(f"Propellant Mass Used: {change_in_mass:.2f} kg")
    #print(f"Final Velocity Vector: {final_speed} m/s")
    #print(f"Final Mass: {final_mass:.2f} kg")

    # semi-major axis of the final orbit
    semi_major_axis_orbit = R_earth + altitude_orbit
    # Orbital period of the final orbit
    T_orbit = 2 * np.pi * np.sqrt(semi_major_axis_orbit**3 / (G * M_earth))
    # Time span for one orbital period
    t_span = (t_start, t_start + T_orbit)

    # Integrate the equations of motion
    solution = solve_ivp(coasting_derivatives,
                         t_span,
                         final_state,
                         max_step=0.1,
                         rtol=1e-8,
                         atol=1e-8)
    # Extract position data
    r = solution.y[0:3]

    # Plot the circular orbit
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[0], r[1], r[2], label='Circular Orbit')
    ax.scatter(0, 0, 0, color='yellow', label='Earth', s=100)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('Final Circular Orbital Trajectory')
    plt.savefig(save_file_path + '/final_circular_orbit.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()
    
    return solution.t, solution.y, final_state