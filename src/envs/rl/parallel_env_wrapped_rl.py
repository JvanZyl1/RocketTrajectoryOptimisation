import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool
from typing import List, Tuple

from src.envs.rl.env_wrapped_rl import rl_wrapped_env

class ParallelRocketEnv:
    def __init__(self, 
                 flight_phase: str = 'subsonic',
                 num_parallel_envs: int = 4):
        self.num_parallel_envs = num_parallel_envs
        self.flight_phase = flight_phase
        
        # Create a pool of workers
        self.pool = Pool(processes=num_parallel_envs)
        
        # Create individual environments
        self.envs = [rl_wrapped_env(flight_phase=flight_phase) 
                    for _ in range(num_parallel_envs)]
        
        # Get state and action dimensions from first env
        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim
        
    def reset(self) -> jnp.ndarray:
        """Reset all environments in parallel"""
        states = self.pool.map(lambda env: env.reset(), self.envs)
        return jnp.array(states)
    
    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[dict]]:
        """Step all environments in parallel with vectorized operations"""
        # Convert JAX array to list of numpy arrays
        actions_list = [np.array(action) for action in actions]
        
        # Step all environments in parallel
        results = self.pool.starmap(
            lambda env, action: env.step(action),
            zip(self.envs, actions_list)
        )
        
        # Unpack results
        next_states, rewards, dones, truncateds, infos = zip(*results)
        
        # Convert to JAX arrays
        next_states = jnp.array(next_states)
        rewards = jnp.array(rewards)
        dones = jnp.array(dones)
        truncateds = jnp.array(truncateds)
        
        return next_states, rewards, dones, truncateds, infos
    
    def close(self):
        """Close all environments and the process pool"""
        self.pool.close()
        self.pool.join()
        for env in self.envs:
            env.close()
    
    def __del__(self):
        """Ensure proper cleanup"""
        self.close() 