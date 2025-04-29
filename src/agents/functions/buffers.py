import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

@partial(jax.jit, static_argnames=('gamma', 'state_dim', 'action_dim', 'n'))
def compute_n_step_single(
    buf: jnp.ndarray,
    gamma: float,
    state_dim: int,
    action_dim: int,
    n: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # indices into a single transition
    rew_i  = state_dim + action_dim
    ns_i   = rew_i + 1
    ned_i  = ns_i + state_dim
    done_i = ned_i

    # reverse first n transitions for backward return
    seq = buf[:n][::-1]

    def backward_step(carry, tr):
        G, next_s = carry
        r        = tr[rew_i]
        s_next   = tr[ns_i:ned_i]
        d        = tr[done_i] > 0.5

        # reset at terminal, else accumulate discounted return
        G      = jnp.where(d, r, r + gamma * G)
        next_s = jnp.where(d, s_next, next_s)
        return (G, next_s), None

    init = (0.0, jnp.zeros(state_dim, dtype=jnp.float32))
    (G, next_state), _ = jax.lax.scan(backward_step, init, seq)

    # whether any of the n transitions was terminal
    done_any = jnp.any(buf[:n, done_i] > 0.5)

    return (
        G.astype(jnp.float32),
        next_state.astype(jnp.float32),
        done_any.astype(jnp.float32),
    )


class PERBuffer:
    def __init__(self,
                 gamma : float,
                 alpha : float,
                 beta : float,
                 beta_decay : float,
                 buffer_size : int,
                 state_dim : int,
                 action_dim : int,
                 trajectory_length : int,
                 batch_size : int):
        # Hyper parameters
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_original = beta
        self.batch_size = batch_size
        self.beta_decay = beta_decay
        self.buffer_size = buffer_size
        self.trajectory_length = trajectory_length

        # Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Buffer
        transition_dim_n_step = state_dim + action_dim + 1 + state_dim + 1      # 4 + 2 + 1 + 4 + 1 = 12 (no td_error)
        self.n_step_buffer = jnp.zeros((trajectory_length, transition_dim_n_step), dtype=jnp.float32)
        self.position = 0
        self.current_size = 0  # Track the actual number of valid entries
        self.priorities = jnp.ones(buffer_size, dtype=jnp.float32) * 1e-10  # Small non-zero value for priorities

        transition_dim = state_dim + action_dim + 1 + state_dim + 1 + 1  # state + action + reward + next_state + done + td_error : buffer
        self.buffer = jnp.zeros((buffer_size, transition_dim), dtype=jnp.float32)

        # Uniform beun fix
        self.uniform_beun_fix_bool = False # BEUN fix
        self.verbose = True  # Control warning messages
        self._last_sampling_mode = False  # Track last sampling mode to reduce warnings
        
    def reset(self):
        self.buffer = jnp.zeros_like(self.buffer)
        self.priorities = jnp.ones_like(self.priorities) * 1e-6
        self.position = 0
        self.current_size = 0
        self.n_step_buffer = jnp.zeros_like(self.n_step_buffer)
        self.beta = self.beta_original

    def __call__(self,
                 rng_key: jax.random.PRNGKey):
        
        if self.uniform_beun_fix_bool:
            if self.verbose and self._last_sampling_mode != self.uniform_beun_fix_bool:
                print("Warning: Using uniform sampling (uniform_beun_fix_bool=True). PER weights will all be 1.0")
            probabilities = jnp.ones(self.buffer_size) / self.buffer_size
        else:
            # Add small epsilon to all priorities to prevent any zeros
            priorities_plus_eps = self.priorities + 1e-6
            # Calculate probabilities with stability fixes
            probabilities = (priorities_plus_eps ** self.alpha) / jnp.sum(priorities_plus_eps ** self.alpha)
            
        indices = jax.random.choice(rng_key,
                                    self.buffer_size,
                                    shape=(self.batch_size,),
                                    p=probabilities)
        samples = self.buffer[indices]
        
        if self.uniform_beun_fix_bool:
            weights = jnp.ones(self.batch_size)
        else:
            # The weight calculation: (p_i * N)^-β
            # Add small epsilon for numerical stability
            weights = (probabilities[indices] * self.buffer_size + 1e-10) ** (-self.beta)
            # Normalize weights to prevent extremely large values
            weights = weights / jnp.max(weights)

        states = samples[:, :self.state_dim]
        actions = samples[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = samples[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
        next_states = samples[:, self.state_dim + self.action_dim + 1:self.state_dim + self.state_dim + self.action_dim + 1]
        dones = samples[:, self.state_dim + self.state_dim + self.action_dim + 1:self.state_dim + self.state_dim + self.action_dim + 2]

        self.beta = jnp.minimum(1.0, self.beta * self.beta_decay)
        
        # Update last sampling mode to minimize warning messages
        self._last_sampling_mode = self.uniform_beun_fix_bool

        return states, actions, rewards, next_states, dones, indices, weights      
        
    def add(self,
            state : jnp.ndarray,
            action : jnp.ndarray,
            reward : float,
            next_state : jnp.ndarray,
            done : bool,
            td_error : float):
        
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        next_state = jnp.asarray(next_state, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.float32)
        td_error = jnp.asarray(td_error, dtype=jnp.float32)
        
        self.n_step_buffer = self.n_step_buffer.at[self.position % self.trajectory_length].set(jnp.concatenate([state,
                                                                            action,
                                                                            jnp.array([reward]),
                                                                            next_state,
                                                                            jnp.array([done])]))

        n_step_reward, _, _ = compute_n_step_single(self.n_step_buffer,
                                                    self.gamma,
                                                    self.state_dim,
                                                    self.action_dim,
                                                    n=self.trajectory_length)
        transition = jnp.concatenate([
            state,
            action,
            jnp.array([n_step_reward]),
            next_state,
            jnp.array([done]),
            jnp.array([td_error])
        ])
        self.buffer = self.buffer.at[self.position].set(transition)
        self.priorities = self.priorities.at[self.position].set(jnp.abs(td_error) + 1e-6)
        self.position = (self.position + 1) % self.buffer_size
        # Update current size counter
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def update_priorities(self,
                          indices: jnp.ndarray,
                          td_errors: jnp.ndarray):
        td_errors = jnp.squeeze(td_errors)
        self.priorities = self.priorities.at[indices].set(jnp.abs(td_errors) + 1e-6)

    def __len__(self):
        # Return actual number of valid entries rather than buffer capacity
        return self.current_size
    
    def capacity(self):
        """Return the maximum capacity of the buffer"""
        return self.buffer_size
        
    def set_uniform_sampling(self, value: bool, verbose=None):
        """Set whether to use uniform sampling (True) or prioritized sampling (False)"""
        # Only display warning if the mode has actually changed
        mode_changed = self.uniform_beun_fix_bool != value
        
        self.uniform_beun_fix_bool = value
        
        # Update verbosity if provided
        if verbose is not None:
            self.verbose = verbose
            
        if self.verbose and mode_changed:
            if value:
                print("PER buffer switched to uniform sampling (weights will be 1.0)")
            else:
                print("PER buffer switched to prioritized sampling")
            
    def is_using_uniform_sampling(self):
        """Return True if buffer is using uniform sampling rather than prioritized sampling"""
        return self.uniform_beun_fix_bool