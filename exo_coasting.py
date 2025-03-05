import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R_earth = 6371e3  # m
mu = 3.986e14  # m^3/s^2
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]

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
                                     target_altitude):
    
    # Mock t_span to cover all events
    t_span = [t_start, t_start+150]
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
    altitude = np.linalg.norm(sol.y[:3], axis=0) - R_earth
    
    # Fix: Properly reshape arrays for cross product calculation
    positions = sol.y[:3]
    velocities = sol.y[3:6]
    vel_rel = np.zeros_like(velocities)
    
    for i in range(positions.shape[1]):
        vel_rel[:, i] = velocities[:, i] - np.cross(w_earth, positions[:, i])
    
    vel_rel_mag = np.linalg.norm(vel_rel, axis=0)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(sol.t, altitude/1000)
    axs[0].set_ylabel('Altitude [km]')
    axs[0].set_xlabel('Time [s]')
    axs[1].plot(sol.t, vel_rel_mag)
    axs[1].set_ylabel('Relative Velocity [m/s]')
    axs[1].set_xlabel('Time [s]')
    axs[2].plot(sol.t, sol.y[6])
    axs[2].set_ylabel('Mass [kg]')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig('results/exo_coasting.png')
    plt.close()

    return sol.t, sol.y, final_state