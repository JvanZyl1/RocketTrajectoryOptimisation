import numpy as np

from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent
from src.envs.gymnasium_wrapper import GymnasiumWrapper

class ascent_wrapped_env(GymnasiumWrapper):
    def __init__(self,
                 sizing_needed_bool: bool = False,
                 print_bool: bool = False):
        env = rocket_model_endo_ascent(sizing_needed_bool = sizing_needed_bool)
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        
        self.state_dim = 5
        self.action_dim = 3

        super().__init__(env, print_bool)

    def augment_action(self, action):
        return action
    
    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        return np.array([x, y, theta, theta_dot, alpha])
