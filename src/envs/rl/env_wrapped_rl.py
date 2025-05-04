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
        pass

    def close(self):
        pass
    
    def __getattr__(self, name):
        return getattr(self.env, name)

class rl_wrapped_env(GymnasiumWrapper):
    def __init__(self,
                 flight_phase: str = 'subsonic',
                 enable_wind: bool = False):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
        self.flight_phase = flight_phase
        env = rocket_environment_pre_wrap(type = 'rl',
                                          flight_phase = flight_phase,
                                          enable_wind = enable_wind)
        self.enable_wind = enable_wind
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.state_dim = 8
            self.action_dim = 2
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state_dim = 2
            self.action_dim = 1
        elif self.flight_phase == 'ballistic_arc_descent':
            self.state_dim = 4
            self.action_dim = 1
        elif self.flight_phase == 're_entry_burn':
            self.state_dim = 9
            self.action_dim = 2

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
        elif self.flight_phase == 'ballistic_arc_descent':
            action_state = np.array([theta, theta_dot, gamma, alpha])
        elif self.flight_phase == 're_entry_burn':
            action_state = np.array([x, y, vx, vy, theta, theta_dot, gamma, alpha, mass])
        action_state /= self.input_normalisation_vals
        return action_state

    def close(self):
        """Close the environment. This is a no-op for the rocket environment."""
        pass  # No cleanup needed for the rocket environment
