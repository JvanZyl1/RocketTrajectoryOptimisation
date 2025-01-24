import numpy as np

# Constants
mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

# Mission Requirements Data
m_pl = 300  # Payload mass [kg]
h_orbit = 700000  # Orbit altitude [m]
lat = 5.2 * np.pi / 180  # Kourou latitude [rad]
a = h_orbit + R_earth  # Semi-major axis [m]
a_max1 = 7 * g0  # First stage maximum acceleration [m/s^2]
a_max2 = 6 * g0  # Second stage maximum acceleration [m/s^2]

# Design Concept Data
N = 2  # Number of stages
eps = [0.10, 0.13]  # Structural Coefficient
Isp = [300, 320]  # Vacuum Specific Impulse [s]

# Simulator 3 DoF Data
A_a = 1  # Reference aerodynamic area [m^2]
A_e = 0.3  # Exhaust nozzle area [m^2]
P_e = 40000  # Nozzle exit pressure [Pa]
M0 = np.array([0.2, 0.5, 0.8, 1.2, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5])  # Mach number
cD0 = np.array([0.27, 0.26, 0.25, 0.5, 0.46, 0.44, 0.41, 0.39, 0.37,
                0.35, 0.33, 0.3, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21])  # Drag coefficient