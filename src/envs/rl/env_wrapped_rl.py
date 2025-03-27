import jax
import numpy as np
import gymnasium as gym
import jax.numpy as jnp

from src.envs.base_environment import rocket_environment_pre_wrap

class GymnasiumWrapper:
    def __init__(self,
                 env: gym.Env):
        self.env = env
    def reset(self):
        state  = self.env.reset()
        processed_state = self._process_state(state)  
        return processed_state
    
    def augment_action(self, action):
        # done in child class
        return action
    
    def augment_state(self, state):
        # done in child class
        return state

    def step(self, action):
        if isinstance(action, jnp.ndarray):
            action = np.array(jax.device_get(action))  # Convert JAX array to NumPy array
        action = self.augment_action(action)
        state, reward, done, truncated, info = self.env.step(action)
        processed_state = self._process_state(state)
        return processed_state, float(reward), bool(done), bool(truncated), info

    def _process_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
        state = self.augment_state(state)
        return state

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)

class rl_wrapped_env(GymnasiumWrapper):
    def __init__(self,
                 sizing_needed_bool: bool = False):
        env = rocket_environment_pre_wrap(sizing_needed_bool = sizing_needed_bool, type = 'rl')
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        
        self.state_dim = 5
        self.action_dim = 3

        super().__init__(env)

    def augment_action(self, action):
        return action
    
    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        return np.array([x, y, theta, theta_dot, alpha])
