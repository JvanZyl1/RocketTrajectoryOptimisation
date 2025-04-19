import jax
import numpy as np
import gymnasium as gym
import jax.numpy as jnp

from src.envs.base_environment import rocket_environment_pre_wrap
from src.envs.utils.input_normalisation import find_input_normalisation_vals

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
                 flight_phase: str = 'subsonic'):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']
        self.flight_phase = flight_phase
        env = rocket_environment_pre_wrap(type = 'rl',
                                          flight_phase = flight_phase)
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.state_dim = 8
            self.action_dim = 2
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state_dim = 8
            self.action_dim = 1

        self.input_normalisation_vals = find_input_normalisation_vals(flight_phase)

        super().__init__(env)
    
    def truncation_id(self):
        return self.env.truncation_id

    def augment_action(self, action):
        return action
    
    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if self.flight_phase in ['subsonic', 'supersonic']:
            action_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
        elif self.flight_phase == 'flip_over_boostbackburn':
            action_state = np.array([theta, theta_dot])
        action_state /= self.input_normalisation_vals
        return action_state
