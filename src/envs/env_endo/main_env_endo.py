import numpy as np
import os
import dill
from src.envs.env_endo.physics_plotter import test_physics_endo_with_plot as test_physics
from src.envs.env_endo.physics_endo import setup_physics_step_endo as setup_physics



class rocket_model_endo_ascent:
    def __init__(self,
                 dt : float):

        self.dt = dt

        self.truncation_id = 0
        self.startup()


    def startup(self):
        self.physics_step = setup_physics(self.dt)
        test_physics(self)
        self.reset()

    def step(self, actions):
        # Physics step
        self.physics_state, self.propellant_mass, dynamic_pressure, info = self.physics_step(self.physics_state,
                                                                                                         actions,
                                                                                                         self.propellant_mass)
        # Augment state
        self.agent_state = self.augment_state(self.physics_state)
        # Truncated function
        truncated = self.truncated_func(dynamic_pressure)
        # Done function
        done = self.done_func()
        # Reward function
        reward = self.reward_func(actions, done, truncated)
        
        if self.print_bool:
            print(f'ACTIONS: {actions}')
            print(f'State: {self.agent_state}, Reward: {reward}, Done: {done}, Truncated: {truncated}')

        info['physics_state'] = self.physics_state
        info['propellant_mass'] = self.propellant_mass
        info['dynamic_pressure'] = dynamic_pressure

        return self.agent_state, reward, done, truncated, info