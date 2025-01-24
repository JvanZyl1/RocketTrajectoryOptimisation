import numpy as np
from scipy.integrate import solve_ivp
from params import rho0, scale_height_endo, R_earth, w_earth, g0, mu
from params import aerodynamic_area, nozzle_exit_area, nozzle_exit_pressure, \
    get_drag_coefficient, specific_impulses_vacuum


def endo_atmospheric_model(h):
    """
    Provides the atmospheric properties for the endo-atmospheric model.
    Returns:
        rho: float - Density [kg/m^3]
        P_a: float - Pressure [Pa]
        a: float - Speed of sound [m/s]
    """
    if h >= 0:
        rho = rho0 * np.exp(-h / scale_height_endo)
        P_a = 101325 * (rho / rho0)
    else:
        rho = rho0
        P_a = 101325
    a = 340.29  # Speed of sound (assumed constant here)
    return rho, P_a, a


def rocket_dynamics(t, state_vector, mass_flow_endo):
    """
    Computes the derivatives of the state vector for the rocket dynamics.
    Args:
        t: float - Current time [s]
        state_vector: ndarray - Current state [x, y, z, vx, vy, vz, m]
        mass_flow_endo: float - Mass flow rate [kg/s]

    Returns:
        ndarray: Derivatives [dx, dy, dz, dvx, dvy, dvz, dm]
    """
    # Extract position, velocity, and mass
    position_vector = state_vector[:3]
    velocity_vector = state_vector[3:6]
    m = state_vector[6]

    # Compute altitude and atmospheric properties
    altitude = np.linalg.norm(position_vector) - R_earth
    density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(altitude)

    print(f"t = {t:.2f}, Altitude = {altitude:.2f} m, Mass = {m:.2f} kg")  # Debugging

    # Compute relative velocity, Mach number, and drag coefficient
    velocity_relative = velocity_vector - np.cross(w_earth, position_vector)
    Mach_number = np.linalg.norm(velocity_relative) / speed_of_sound
    drag_coefficient = get_drag_coefficient(Mach_number)

    # Compute thrust and drag
    thrust = specific_impulses_vacuum[0] * g0 * mass_flow_endo + \
             (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area
    drag = 0.5 * density * np.linalg.norm(velocity_relative)**2 * aerodynamic_area * drag_coefficient

    # Compute derivatives
    r_dot = velocity_vector
    v_dot = (
        -mu / np.linalg.norm(position_vector)**3 * position_vector  # Gravity
        + thrust / m * position_vector / np.linalg.norm(position_vector)  # Thrust
        - drag / m * position_vector / np.linalg.norm(position_vector)  # Drag
    )
    change_in_mass = -mass_flow_endo  # Mass flow rate

    return np.concatenate((r_dot, v_dot, [change_in_mass]))


def altitude_event(t, state_vector, target_altitude=100):
    """
    Event function to stop integration when a target altitude is reached.
    """
    altitude = np.linalg.norm(state_vector[:3]) - R_earth
    print(f"Altitude Event Check: Altitude = {altitude:.2f}, Target = {target_altitude}")  # Debugging
    return altitude - target_altitude


def mass_event(t, state_vector, minimum_mass_end_of_burn=300000):
    """
    Event function to stop integration when the mass reaches a specified value.
    """
    m = state_vector[6]
    print(f"Mass Event Check: Mass = {m:.2f}, Minimum = {minimum_mass_end_of_burn}")  # Debugging
    return m - minimum_mass_end_of_burn


# Configure event properties
altitude_event.terminal = True
altitude_event.direction = 1  # Positive crossing

mass_event.terminal = True
mass_event.direction = -1  # Negative crossing


# Initial conditions
tspan = (0, 5000)  # Integration time span
x0 = np.array([R_earth + 0, 0, 0, 0, 0, 0, 500000])  # Initial position, velocity, and mass
mass_flow_endo = 100  # Example mass flow rate

initial_altitude = np.linalg.norm(x0[:3]) - R_earth
print(f"Initial Altitude: {initial_altitude} m")
initial_mass = x0[6]
print(f"Initial Mass: {initial_mass} kg")

# Solve the system
solution = solve_ivp(
    lambda t, y: rocket_dynamics(t, y, mass_flow_endo),
    tspan,
    x0,
    events=[lambda t, y: altitude_event(t, y, target_altitude=100),
            lambda t, y: mass_event(t, y, minimum_mass_end_of_burn=300000)],
    atol=1e-8,
    rtol=1e-8
)

# Extract results
t = solution.t
x = solution.y.T
event_times = solution.t_events
event_states = solution.y_events

# Compute altitude over time
altitudes = np.linalg.norm(x[:, :3], axis=1) - R_earth

# Display results
print("Integration completed.")
print(f"Event times: {event_times}")
print(f"Event states: {event_states}")
