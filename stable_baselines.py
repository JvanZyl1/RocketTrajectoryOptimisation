import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC

from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent

class endo_ascent_wrapped_EA(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = rocket_model_endo_ascent()
        self.initial_mass = self.env.reset()[-2]
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Assuming state space bounds for [x, y, theta, theta_dot, alpha]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.pi/2]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi/2]),
            dtype=np.float32
        )

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        
        # Handle tensors by detaching them before converting to numpy
        if isinstance(x, torch.Tensor):
            return torch.tensor([x.detach(),
                                 y.detach(),
                                 theta.detach(),
                                 theta_dot.detach(),
                                 alpha.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, theta, theta_dot, alpha])
    
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

env = endo_ascent_wrapped_EA()

model = SAC("MlpPolicy",
            env,
            verbose=1)

model.learn(total_timesteps=1e8,
            log_interval=100)

# ep_len_mean : Average length of episodes
# ep_rew_mean : Average reward of episodes
# Episodes : Number of episodes
# fps : Frames per second
# time_elapsed : Time taken to train
# total_timesteps : Total time steps
# actor_loss : Actor loss
# ent_coef : Entropy coefficient
# ent_coef_loss : Entropy coefficient loss
# learning_rate : Learning rate
# n_updates : Number of updates

model.save("sac_endo_ascent")