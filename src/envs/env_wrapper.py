import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import jax
import math

class EnvWrapper_Skeleton:
    """
    A wrapper for preprocessing environment interactions to ensure consistent inputs
    for the replay buffer.
    """
    def __init__(self,
                 env: gym.Env,
                 print_bool: bool = False,
                 state_max: np.array = None):
        self.env = env
        self.print_bool = print_bool
        self.state_max = state_max if state_max is not None else np.ones(env.observation_space.shape)
    def reset(self):
        """
        Reset the environment and preprocess the initial state.
        """
        state, _ = self.env.reset()
        processed_state = self._process_state(state)  
        return processed_state

    def step(self, action):
        """
        Take a step in the environment and preprocess the outputs.
        """
        # Ensure action is a numpy array to avoid ambiguity in conditionals
        if isinstance(action, jnp.ndarray):
            action = np.array(jax.device_get(action))  # Convert JAX array to NumPy array
        state, reward, done, truncated, info = self.env.step(action)
        processed_state = self._process_state(state)
        return processed_state, float(reward), bool(done), bool(truncated), info

    def _process_state(self, state):
        """
        Preprocess the state returned by the environment.
        Handles tuples by extracting the first element or other necessary parts.
        """
        if isinstance(state, tuple):
            state = state[0]
        state = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
        state = state / (self.state_max + 1e-8)
        return state

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    

class EnvWrapper_VerticalRising(EnvWrapper_Skeleton):
    def __init__(self,
                 env: gym.Env,
                 print_bool: bool = False):
        x_max = 20
        y_max = 300
        vx_max = 10
        vy_max = 30
        theta_max = math.radians(100)
        theta_dot_max = math.radians(1)
        gamma_max = math.radians(100)
        alpha_max = math.radians(10)
        mass_max = env.rocket_configuration['stage_masses_dict']['initial_mass']   
        altitude_error_max = y_max
        target_altitude_max = y_max
        state_max = np.array([x_max, y_max, vx_max, vy_max, theta_max, theta_dot_max, gamma_max, alpha_max, mass_max, altitude_error_max, target_altitude_max])
        super().__init__(env, print_bool, state_max)