import numpy as np
from scipy.interpolate import interp1d
import math

# Constants
mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]
scale_height_endo = 8500  # Scale height for endo-atmospheric model [m]
rho0 = 1.225  # Sea level density [kg/m^3]

# Mission Requirements Data
payload_mass = 300                          # Payload mass [kg]
mass_fairing = 100                          # Fairing mass [kg]

altitude_orbit = 700000                     # Orbit altitude [m]
semi_major_axis = altitude_orbit + R_earth  # Semi-major axis [m]

accel_max_first_stage = 7 * g0              # First stage maximum acceleration [m/s^2]
accel_max_second_stage = 6 * g0             # Second stage maximum acceleration [m/s^2]

# Design Concept Data
number_of_stages = 2                        # Number of stages
structural_coefficients = [0.10, 0.13]      # Structural Coefficient
specific_impulses_vacuum = [300, 320]       # Vacuum Specific Impulse [s]
maximum_accelerations = [accel_max_first_stage, accel_max_second_stage]  # Maximum acceleration for each stage [m/s^2]

# Simulator 3 DoF Data
aerodynamic_area = 1                        # Reference aerodynamic area [m^2]
nozzle_exit_area = 0.3                      # Exhaust nozzle area [m^2]
nozzle_exit_pressure = 40000                # Nozzle exit pressure [Pa]
mach_number_array = np.array([0.2, 0.5, 0.8, 1.2, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5])  # Mach number
cd_array = np.array([0.27, 0.26, 0.25, 0.5, 0.46, 0.44, 0.41, 0.39, 0.37,
                0.35, 0.33, 0.3, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21])  # Drag coefficient
# Make a function which can be called to get the drag coefficient, is a callable function
drag_coefficient_function = interp1d(
    mach_number_array,
    cd_array,
    kind='linear',  # Linear interpolation
    fill_value='extrapolate'  # Allow extrapolation for Mach numbers outside the range
)
def get_drag_coefficient(mach_number):
    return drag_coefficient_function(mach_number)


# Initial conditions
latitude = 5.2 * np.pi / 180                                                                          # Kourou latitude [rad] - launch altitude
position_vector_initial = np.array([R_earth * np.cos(latitude), 0, R_earth * np.sin(latitude)])       # Initial position vector [m]
unit_position_vector_initial = position_vector_initial / np.linalg.norm(position_vector_initial)      # Initial position unit vector
east_vector = np.cross([0, 0, 1], unit_position_vector_initial)                                       # East vector [m]
unit_east_vector = east_vector / np.linalg.norm(east_vector)                                          # East unit vector
velocity_vector_initial = np.cross(w_earth, position_vector_initial)                                  # Initial velocity vector [m/s]
unit_velocity_vector_initial = velocity_vector_initial / np.linalg.norm(velocity_vector_initial)      # Initial velocity unit vector
initial_conditions_dictionary = {
    "initial_position_vector": position_vector_initial,
    "initial_position_unit_vector": unit_position_vector_initial,
    "east_vector": east_vector,
    "east_unit_vector": unit_east_vector,
    "initial_velocity_vector": velocity_vector_initial,
    "initial_velocity_unit_vector": unit_velocity_vector_initial
}

# Variables which could maybe optimise.
target_altitude_vertical_rising = 100.0     # target altitude [m]
kick_angle = math.radians(0.1)              # kick angle [deg]
target_altitude_gravity_turn = 1160000.0    # target altitude [m], maximum altitude for gravity turn
coasting_time = 5.0                         # coasting time [s]