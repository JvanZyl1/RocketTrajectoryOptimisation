import numpy as np
from scipy.interpolate import interp1d
import math


physical_constants = {
    'mu': 398602 * 1e9,                                 # Gravitational parameter [m^3/s^2]
    'R_earth': 6378137,                                 # Earth radius [m]
    'w_earth': np.array([0, 0, 2 * np.pi / 86164]),     # Earth angular velocity [rad/s]
    'g0': 9.80665,                                      # Gravity constant on Earth [m/s^2]
    'scale_height_endo': 8500,                          # Scale height for endo-atmospheric model [m]
    'rho0': 1.225,                                      # Sea level density [kg/m^3]
    'M_earth': 5.972e24,                                # Earth mass [kg]
    'G': 6.67430e-11                                    # Gravitational constant [m^3/kg/s^2]
}

mu = physical_constants['mu']
R_earth = physical_constants['R_earth']
w_earth = physical_constants['w_earth']
g0 = physical_constants['g0']
scale_height_endo = physical_constants['scale_height_endo']
rho0 = physical_constants['rho0']
M_earth = physical_constants['M_earth']
G = physical_constants['G']