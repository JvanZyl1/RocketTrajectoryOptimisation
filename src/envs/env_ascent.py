from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent
from src.envs.env_wrapper import EnvWrapper_Skeleton

import math
import csv
import numpy as np


class ascent_wrapped_env(EnvWrapper_Skeleton):
    def __init__(self,
                 sizing_needed_bool: bool = False,
                 print_bool: bool = False):
        env = rocket_model_endo_ascent(sizing_needed_bool = sizing_needed_bool)
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        x_max = 20000
        y_max = 50000
        vx_max = 2000
        theta_dot_max = math.radians(5)
        gamma_max = math.radians(100)
        # Read sizing results
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        mass_propellant_initial = float(sizing_results['Propellant mass stage 1 (ascent)'])*1000  # mass [kg]

        state_max = np.array([x_max,
                              y_max])
                              #vx_max, 
                              #theta_dot_max,
                              #gamma_max,
                              #mass_propellant_initial])
        
        self.state_dim = 2
        self.action_dim = 2

        super().__init__(env, print_bool, state_max)

    def augment_action(self, action):
        # Action is : gimbal/x-throttle, throttle, rcs force, acs angle L, acs angle R
        # Action used here is: gimbal/x-throttle, throttle
        return action
    
    def augment_state(self, state):
        # x, y, vx, gamma, mass propellant, theta_dot
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        return np.array([x, y])
