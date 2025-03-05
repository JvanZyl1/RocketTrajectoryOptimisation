import numpy as np

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]

def final_mass_compute(final_state, v_exhaust, target_altitude):
    pos = final_state[0:3]
    vel = final_state[3:6]
    m = final_state[6]

    # Orbital parameters
    E_temp = np.linalg.norm(pos) * np.linalg.norm(vel)**2 / mu       # Specific energy
    a = np.linalg.norm(pos) / (2 - E_temp)                           # Semi-major axis

    ste = np.dot(pos, vel) * np.linalg.norm(np.cross(pos, vel)) / (np.linalg.norm(pos) * mu)        
    cte = (np.linalg.norm(np.cross(pos,vel)))**2 / (mu * np.linalg.norm(pos)) - 1.0         
    e = np.sqrt(ste**2 + cte**2)                            # Eccentricity

    # Current orbit
    E = -mu/(2*a)                   # Specific energy
    r_a = a*(1 + e)                 # Radius at apogee
    r_p = a*(1 - e)                 # Radius at perigee
    V = np.sqrt(2*(E + mu/r_a))     # Arrival velocity

    # Circular parameters
    a_c = target_altitude + R_earth
    V_c = np.sqrt(mu/a_c)           # Circular velocity

    # Delta V
    delta_v = V_c - V

    m_c = m * np.exp(-delta_v/v_exhaust)
    return m_c