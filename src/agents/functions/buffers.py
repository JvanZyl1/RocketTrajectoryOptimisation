import jax.numpy as jnp
from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax

@partial(jax.jit, static_argnames=['gamma', 'state_dim', 'action_dim'])
def compute_n_step_single(
    n_step_buffer: jnp.ndarray,
    gamma: float,
    state_dim: int,
    action_dim: int
) -> Tuple[float, jnp.ndarray, bool]:
    """
    Compute n-step returns for the oldest transition in a single n-step buffer.

    Args:
        n_step_buffer (jnp.ndarray): Buffer containing transitions for a single trajectory.
        gamma (float): Discount factor.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.

    Returns:
        Tuple containing:
            reward (float): n-step discounted reward.
            next_state (jnp.ndarray): State after n steps.
            done (bool): Done flag after n steps.
    """

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

    # Scan from oldest to newest (do not reverse)
    transitions = n_step_buffer[:-1]

    # Apply lax.scan
    (reward, next_state, done), _ = jax.lax.scan(step, carry, transitions)

    return reward, next_state, done

#@partial(jax.jit, static_argnames=['gamma', 'state_dim', 'action_dim', 'buffer_size'])
def add_to_buffer(buffer: jnp.ndarray,
                  priorities: jnp.ndarray,
                  position: int,
                  state: jnp.ndarray,
                  action: jnp.ndarray,
                  reward: float,
                  next_state: jnp.ndarray,
                  done: bool,
                  td_error: float,
                  n_step_buffer: jnp.ndarray,
                  gamma: float,
                  state_dim: int,
                  action_dim: int,
                  buffer_size: int)-> Tuple[jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    """
    Adds a transition to the buffer with priorities, circular indexing, and n-step update.
    """
    # Ensure correct array format
    state = jnp.asarray(state, dtype=jnp.float32)
    action = jnp.asarray(action, dtype=jnp.float32)
    next_state = jnp.asarray(next_state, dtype=jnp.float32)
    done = jnp.asarray(done, dtype=jnp.float32)
    td_error = jnp.asarray(td_error, dtype=jnp.float32)

    # Due to MARL
    if state.ndim == 2:
        state = jnp.squeeze(state)
    if action.ndim == 2:
        action = jnp.squeeze(action)
    if action.ndim == 0:
        action = jnp.expand_dims(action, axis=0)
    if reward.ndim == 1:
        reward = reward[0]
    if next_state.ndim == 2:
        next_state = jnp.squeeze(next_state)
    if done.ndim == 1:
        done = done[0]    

    
    # Update n-step buffer  
    n_step_buffer = n_step_buffer.at[-1].set(
        jnp.concatenate([state, action, jnp.array([reward]), next_state, jnp.array([done])])
    )
    
    # Compute n-step reward
    n_step_reward, _, _ = compute_n_step_single(n_step_buffer, gamma, state_dim, action_dim)
    n_step_reward = jnp.asarray(n_step_reward, dtype=jnp.float32)

    # Create the new transition for the main buffer
    transition = jnp.concatenate([
        state, action, jnp.array([n_step_reward]), next_state, jnp.array([done]), jnp.array([td_error])
    ])
    
    # Update buffer with circular indexing
    buffer = buffer.at[position].set(transition)

    # Update priorities
    priorities = priorities.at[position].set(jnp.abs(td_error) + 1e-6)

    # Update position (circular indexing)
    position = (position + 1) % buffer_size

    return buffer, priorities, position, n_step_buffer

@partial(jax.jit, static_argnames=['alpha'])
def probability_calculation(alpha : float,
                            priorities : jnp.ndarray) -> jnp.ndarray:
    probabilities = (priorities ** alpha) / jnp.sum(priorities ** alpha)
    return probabilities

@partial(jax.jit, static_argnames=['state_dim', 'action_dim'])
def sample_select(sample: jnp.ndarray,
                   state_dim: int,
                   action_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Transition format: [state_dim, action_dim, reward, state_dim, done, td_error]
    states = sample[:state_dim]                                                                 # States is size state_dim
    actions = sample[state_dim:state_dim + action_dim]                                          # Actions is size action_dim                         
    reward = sample[state_dim + action_dim]                                                     # Reward is size 1
    next_states = sample[state_dim + action_dim + 1:state_dim + state_dim + action_dim + 1]     # Next states is size state_dim
    done = sample[state_dim + state_dim + action_dim + 1]                                       # Done is size 1

    return states, actions, reward, next_states, done

@partial(jax.jit, static_argnames=['alpha', 'beta_decay', 'state_dim', 'action_dim', 'batch_size'])
def sample_from_buffer(buffer: jnp.ndarray,
                       priorities: jnp.ndarray,
                       rng_key: jax.random.PRNGKey,
                       alpha : float,
                       beta_decay : float,
                       state_dim : int,
                       action_dim : int,
                       beta : float,
                       batch_size : int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    probabilities = probability_calculation(alpha, priorities)
    indices = jax.random.choice(rng_key,
                                  len(priorities),
                                  shape=(batch_size,),
                                  p=probabilities)
    # Gather sampled transitions
    samples = buffer[indices]

    # Compute importance-sampling weights
    weights = (probabilities[indices] * len(buffer)) ** (-beta)
    weights /= jnp.max(weights)  # Normalize weights

    # Unpack transitions using static slice index
    states = samples[:, :state_dim]
    actions = samples[:, state_dim:state_dim + action_dim]
    rewards = samples[:, state_dim + action_dim:state_dim + action_dim + 1]
    next_states = samples[:, state_dim + action_dim + 1:state_dim + state_dim + action_dim + 1]
    dones = samples[:, state_dim + state_dim + action_dim + 1:state_dim + state_dim + action_dim + 2]

    beta = jnp.minimum(1.0, beta + beta_decay)
    return states, actions, rewards, next_states, dones, indices, weights, beta

class PERInference:
    def __init__(self,
                 gamma: float = 0.99,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_decay: float = 0.001,
                 buffer_size: int = 100,
                 state_dim: int = 4,
                 action_dim: int = 2,
                 trajectory_length: int = 5,
                 batch_size: int = 32):
        """
        Initialize the PERInference class for Prioritized Experience Replay.

        Args:
            gamma (float): Discount factor for n-step return calculation.
            alpha (float): Priority exponent for PER.
            beta (float): Importance-sampling weight exponent.
            beta_decay (float): Increment for beta per step.
            buffer_size (int): Maximum size of the replay buffer.
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            trajectory_length (int): Number of steps for n-step return calculation.
        """
        # Static params
        static_config = {
            'gamma': gamma,
            'alpha': alpha,
            'buffer_size': buffer_size,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'beta_decay': beta_decay,
            'batch_size': batch_size
        }
        static_config['beta_original'] = beta

        self.buffer_size = buffer_size # So is callable

        self.static_config = static_config
        # Dynamic params
        self.beta = beta

        # Initialize components
        transition_dim = state_dim + action_dim + 1 + state_dim + 1 + 1  # state + action + reward + next_state + done + td_error : buffer
        transition_dim_n_step = state_dim + action_dim + 1 + state_dim + 1      # 4 + 2 + 1 + 4 + 1 = 12 (no td_error)
        self.buffer = jnp.zeros((buffer_size, transition_dim), dtype=jnp.float32)
        self.priorities = jnp.ones(buffer_size, dtype=jnp.float32) * 1e-6  # Small non-zero value for priorities
        self.position = 0
        self.n_step_buffer = jnp.zeros((trajectory_length, transition_dim_n_step), dtype=jnp.float32)

    def __call__(self, rng_key: jax.random.PRNGKey):
        states, actions, rewards, next_states, dones, indices, weights, self.beta = sample_from_buffer(
            buffer=self.buffer,
            priorities=self.priorities,
            rng_key=rng_key,
            alpha = self.static_config['alpha'],
            beta_decay = self.static_config['beta_decay'],
            state_dim = self.static_config['state_dim'],
            action_dim = self.static_config['action_dim'],
            beta = self.beta,
            batch_size = self.static_config['batch_size']
        )
        return states, actions, rewards, next_states, dones, indices, weights
    
    def add(self,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: float,
            next_state: jnp.ndarray,
            done: bool,
            td_error: float):
        self.buffer, self.priorities, self.position, self.n_step_buffer = add_to_buffer(
            buffer=self.buffer,
            priorities=self.priorities,
            position=self.position,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            td_error=td_error,
            n_step_buffer=self.n_step_buffer,
            gamma = self.static_config['gamma'],
            state_dim = self.static_config['state_dim'],
            action_dim = self.static_config['action_dim'],
            buffer_size = self.static_config['buffer_size']
        )

    def update_priorities(self,
                          indices: jnp.ndarray,
                          td_errors: jnp.ndarray):
        td_errors = jnp.squeeze(td_errors)
        self.priorities = self.priorities.at[indices].set(jnp.abs(td_errors) + 1e-6)

    def reset(self):
        self.buffer = jnp.zeros_like(self.buffer)
        self.priorities = jnp.ones_like(self.priorities) * 1e-6
        self.position = 0
        self.n_step_buffer = jnp.zeros_like(self.n_step_buffer)
        self.beta = self.static_config['beta_original']

    # LENGTH OF BUFFER
    def __len__(self):
        return len(self.buffer)