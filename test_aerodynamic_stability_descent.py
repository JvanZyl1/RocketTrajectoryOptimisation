import math

from src.envs.utils.aerodynamic_coefficients import rocket_CD_compiler, rocket_CL_compiler
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

class AerodynamicStabilityDescent:
    def __init__(self):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.CD_func = lambda M, alpha_rad: rocket_CD_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        self.CL_func = lambda M, alpha_rad: rocket_CL_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        

    def step(self):
