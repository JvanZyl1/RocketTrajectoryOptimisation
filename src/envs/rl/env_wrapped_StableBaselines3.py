import os
import math
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.envs.base_environment import rocket_environment_pre_wrap

class wrapped_env_StableBaselines3(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = rocket_environment_pre_wrap(type = 'rl')
        self.initial_mass = self.env.reset()[-3]
        initial_mass_propellant = self.env.reset()[-2]

        mass_low = (self.initial_mass - initial_mass_propellant)/self.initial_mass

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Define observation space with explicit float32 dtype
        self.observation_space = gym.spaces.Box(
                      #     x       y         vx      vy         theta     theta_dot      gamma                alpha    
            low=np.array([-100,   -1000,   -100,     -10,           0,    -np.pi/2,          0,   -math.radians(50)   ], dtype=np.float32),
            high=np.array([35000, 55000,    500,     800,    np.pi*3/2,     np.pi/2,  np.pi*3/2,    math.radians(50)   ], dtype=np.float32),
            dtype=np.float32
        )

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        
        # Handle tensors by detaching them before converting to numpy
        if isinstance(x, torch.Tensor):
            return torch.tensor([x.detach(),
                                 y.detach(),
                                 vx.detach(),
                                 vy.detach(),
                                 theta.detach(),
                                 theta_dot.detach(),
                                 gamma.detach(),
                                 alpha.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, vx, vy, theta, theta_dot, gamma, alpha])
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action_detached = action.detach().numpy()
        else:
            action_detached = action
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        state = self.augment_state(state)
        return state, {}  # Gymnasium requires returning a dict as info


def compile_StableBaselines3_env(model_name: str,
                                 norm_obs = True,
                                 norm_reward = False):
    log_dir = f'data/agent_saves/StableBaselines3/{model_name}/logs'
    os.makedirs(log_dir, exist_ok=True)
    env = wrapped_env_StableBaselines3()
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    return env