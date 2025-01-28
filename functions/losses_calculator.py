import numpy as np
from functions.params import R_earth, G, w_earth, get_drag_coefficient, aerodynamic_area, nozzle_exit_area
from functions.endo_atmosphere_vertical_rising import endo_atmospheric_model

def losses_calculator(t,
                      state_vector,
                      dt,
                      endo_atmosphere_bool = True):
    '''
    state_vector: [x, y, z, vx, vy, vz, m]

    reference frame : Inertial Equatorial reference system:
    - X-axis : through meridian passing through launch site.
    - Y-axis : as consequence.
    - Z-axis : through North pole.

    Losses calculator for the rocket trajectory optimisation problem.
    - Gravity losses : \int_{0}^{t} g * sin(gamma) * dt
    - Drag losses : \int_{0}^{t} D/m * dt
    - Pressure losses : \int_{0}^{t} p_a * A_e / m * dt
    - Steering losses : ...
    '''
    # Unpack state vector
    x, y, z, vx, vy, vz, m = state_vector
    position = state_vector[:3]                     # position vector [m]
    vel = state_vector[3:6]                         # velocity vector [m/s]

    gamma = np.arctan(np.sqrt(vx**2 + vy**2) / vz)  # flight path angle [rad]
    altitude = np.linalg.norm([x, y, z]) - R_earth  # altitude [m]
    g = G * (R_earth / (R_earth + altitude))**2     # gravity acceleration [m/s^2]
    Lg = g * np.sin(gamma) * dt                     # gravity losses [m/s^2]
    
    # Steering losses
    # Unsure
    Ls = 0

    if endo_atmosphere_bool:
        # ENDO ONLY
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(altitude) # air density, atmospheric pressure, speed of sound
        vel_rel = vel - np.cross(w_earth, position)         # relative velocity vector [m/s]
        mach = np.linalg.norm(vel_rel) / speed_of_sound     # Mach number [-]
        cd = get_drag_coefficient(mach)                     # drag coefficient [-]
        drag = 0.5 * air_density * (np.linalg.norm(vel_rel)**2) * aerodynamic_area * cd # drag force [N]
        Ld = drag / m * dt                                  # drag losses [m/s^2]

        # ENDO ONLY
        Lp = atmospheric_pressure * nozzle_exit_area / m * dt # pressure losses [m/s^2]
        losses = Lg + Ld + Lp + Ls

    else:
        # EXO ONLY
        losses = Lg + Ls
    return losses

def losses_over_states(states,
                       times,
                       endo_atmosphere_bool = True):
    dt_array = np.diff(times)
    # vertical_rising_states: (7, 1004)
    # vertical_rising_time: (1004,)
    # with 7 states and 1004 time steps
    total_losses = 0
    for i in range(len(times)):
        total_losses += losses_calculator(times[i], states[:, i], dt_array[i], endo_atmosphere_bool)
    return total_losses
