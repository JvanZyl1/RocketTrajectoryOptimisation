import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

### BUFFER ###
@partial(jax.jit, static_argnames=['gamma', 'state_dim', 'action_dim'])
def compute_n_step_single(
    n_step_buffer: jnp.ndarray,
    gamma: float,
    state_dim: int,
    action_dim: int
) -> Tuple[float, jnp.ndarray, bool]:

    # Calculate dynamic indices based on state_dim and action_dim
    reward_idx = state_dim + action_dim
    next_state_start = reward_idx + 1
    next_state_end = next_state_start + state_dim
    done_idx = next_state_end

    def step(carry, transition):
        reward_accum, next_state, done_accum = carry
        r = transition[reward_idx]
        n_s = transition[next_state_start:next_state_end]
        d = transition[done_idx] > 0.5  # Convert to bool

        # Accumulate reward only if not done
        new_reward_accum = jax.lax.cond(
            done_accum,
            lambda _: reward_accum,
            lambda _: r + gamma * reward_accum,
            operand=None
        )

        # Update done flag
        new_done_accum = jax.lax.cond(
            done_accum,
            lambda _: done_accum,
            lambda _: d,
            operand=None
        )

        # Update next_state only if done
        new_next_state = jax.lax.cond(
            d,
            lambda _: n_s,
            lambda _: next_state,
            operand=None
        )

        return (new_reward_accum, new_next_state, new_done_accum), None

    # Initialize carry with zero reward, zero next_state, done=False
    initial_reward = 0.0
    initial_next_state = jnp.zeros(state_dim, dtype=jnp.float32)
    initial_done = False
    carry = (initial_reward, initial_next_state, initial_done)
    transitions = n_step_buffer[:-1]
    (reward, next_state, done), _ = jax.lax.scan(step, carry, transitions)

    return jnp.asarray(reward, dtype=jnp.float32), jnp.asarray(next_state, dtype=jnp.float32), jnp.asarray(done, dtype=jnp.float32)

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
        self.priorities = jnp.ones(buffer_size, dtype=jnp.float32) * 1e-10  # Small non-zero value for priorities

        transition_dim = state_dim + action_dim + 1 + state_dim + 1 + 1  # state + action + reward + next_state + done + td_error : buffer
        self.buffer = jnp.zeros((buffer_size, transition_dim), dtype=jnp.float32)
        
    def reset(self):
        self.buffer = jnp.zeros_like(self.buffer)
        self.priorities = jnp.ones_like(self.priorities) * 1e-6
        self.position = 0
        self.n_step_buffer = jnp.zeros_like(self.n_step_buffer)
        self.beta = self.beta_original

    def __call__(self,
                 rng_key: jax.random.PRNGKey):
        
        probabilities = (self.priorities ** self.alpha) / jnp.sum(self.priorities ** self.alpha)
        indices = jax.random.choice(rng_key,
                                    self.buffer_size,
                                    shape=(self.batch_size,),
                                    p=probabilities)
        samples = self.buffer[indices]
        weights = (probabilities[indices] * self.buffer_size) ** (-self.beta)
        weights /= jnp.max(weights)

        states = samples[:, :self.state_dim]
        actions = samples[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = samples[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + 1]
        next_states = samples[:, self.state_dim + self.action_dim + 1:self.state_dim + self.state_dim + self.action_dim + 1]
        dones = samples[:, self.state_dim + self.state_dim + self.action_dim + 1:self.state_dim + self.state_dim + self.action_dim + 2]

        self.beta = jnp.minimum(1.0, self.beta + self.beta_decay)

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
        
        self.n_step_buffer = self.n_step_buffer.at[-1].set(jnp.concatenate([state,
                                                                            action,
                                                                            jnp.array([reward]),
                                                                            next_state,
                                                                            jnp.array([done])]))

        n_step_reward, _, _ = compute_n_step_single(self.n_step_buffer,
                                                    self.gamma,
                                                    self.state_dim,
                                                    self.action_dim)
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

    def update_priorities(self,
                          indices: jnp.ndarray,
                          td_errors: jnp.ndarray):
        td_errors = jnp.squeeze(td_errors)
        self.priorities = self.priorities.at[indices].set(jnp.abs(td_errors) + 1e-6)

    def __len__(self):
        return self.buffer_size