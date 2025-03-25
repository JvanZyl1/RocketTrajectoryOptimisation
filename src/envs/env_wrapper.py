import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import jax
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
    def reset(self):
        """
        Reset the environment and preprocess the initial state.
        """
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
        """
        Take a step in the environment and preprocess the outputs.
        """
        # Ensure action is a numpy array to avoid ambiguity in conditionals
        if isinstance(action, jnp.ndarray):
            action = np.array(jax.device_get(action))  # Convert JAX array to NumPy array
        action = self.augment_action(action)
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
        state = self.augment_state(state)
        return state

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    

