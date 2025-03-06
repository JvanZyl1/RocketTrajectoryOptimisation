from typing import Tuple
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

@jax.jit
def gaussian_likelihood(actions: jnp.ndarray,
                        mean: jnp.ndarray,
                        std: jnp.ndarray) -> jnp.ndarray:
    '''
    Compute the log likelihood of actions under a Gaussian distribution.

    params:
    actions: Sampled actions [jnp.ndarray]
    mean: Mean of the Gaussian distribution [jnp.ndarray]
    std: Standard deviation of the Gaussian distribution [jnp.ndarray]

    returns:
    log_prob: Log likelihood of the actions [jnp.ndarray]

    notes:
    The Gaussian log likelihood is given by:
    log_prob = -0.5 * ((actions - mean)^2 / (std^2) + 2 * log(std) + log(2 * pi))
    '''
    log_prob = -0.5 * (
        ((actions - mean) ** 2) / (std ** 2)  # Quadratic term
        + 2 * jnp.log(std)  # Log scale normalization
        + jnp.log(2 * jnp.pi)  # Constant factor
    )
    return log_prob.sum(axis=-1)  # Sum over the action dimensions

@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """
    Clips gradients by their global norm to ensure stability.

    Args:
        grads: Gradients to clip, as a PyTree.
        max_norm: Maximum norm for the gradients.

    Returns:
        Clipped gradients with the same structure as the input.
    """
    # Compute the global norm
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    # Compute scaling factor
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    # Scale gradients
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads

def sample_actions(states: jnp.ndarray,
                   actor_params: jnp.ndarray,
                   normal_dist: jnp.ndarray,
                   actor: nn.Module,
                   std_min: float,
                   std_max: float,
                   print_bool: bool) -> jnp.ndarray:
    """
    Samples actions from the current actor given states.

    Args:
        states: jnp.ndarray.

    Returns:
        actions: Clipped actions sampled from the policy [jnp.ndarray].
    """
    # Ensure inputs are jax.numpy arrays

    # Compute mean and standard deviation from the actor
    mean, std_0_1 = actor.apply(actor_params, states)
    std = (std_max - std_min) * std_0_1 + std_min  # Fix to [0.01, 0.1]

    # Sample actions using normal distribution
    actions = normal_dist * std + mean

    # Clip actions to the valid action range
    # TEST FOR MULTIPLE ACTIONS
    actions = jnp.squeeze(actions)

    return actions, mean, std

def calculate_td_error(states,
                       actions,
                       rewards,
                       next_states,
                       dones,
                       temperature: float,
                       gamma: float,
                       critic_params: jnp.ndarray,
                       critic_target_params: jnp.ndarray,
                       critic: nn.Module,
                       next_log_policy: jnp.ndarray,
                       print_bool: bool) -> jnp.ndarray:
    
    if actions.ndim == 0:
        actions = jnp.expand_dims(actions, axis=0)

    # This is for a unimodal Gaussian Double critic

    q1, q2 = critic.apply(critic_params, states, actions) # (mean, std)
    # `not_done` is 1 for non-terminal states, used to mask terminal states in updates
    not_done = 1 - dones

    # Apply the target critic to compute Q-value logits for the next state-action pairs
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, actions)

    # Entropy term
    next_log_policy = jnp.expand_dims(next_log_policy, axis=1)
    entropy_term = temperature * next_log_policy

    # Compute the target Q values
    next_q_mean = jnp.minimum(next_q1, next_q2)
    
    td_target = rewards + gamma * not_done * (next_q_mean - entropy_term)

    # Compute the TD errors : inverse standard deviation weighting
    td_error1 = td_target - q1
    td_error2 = td_target - q2
    td_errors = 0.5 * (td_error1**2 + td_error2**2)
    
    return td_errors


def critic_update(critic_optimizer,
                  calculate_td_error_fn,
                  states : jnp.ndarray,
                  actions : jnp.ndarray,
                  rewards : jnp.ndarray,
                  next_states : jnp.ndarray,
                  dones : jnp.ndarray,
                  weights: jnp.ndarray,
                  # Critic
                  critic_params : jnp.ndarray,
                  critic_opt_state: jnp.ndarray,
                  critic_grad_max_norm: float,
                  critic_target_params: jnp.ndarray,
                  # Other
                  temperature: float,
                  next_log_policy: jnp.ndarray,
                  print_bool: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Update the critic network, which are Gaussian distributions
    # Essentially minimizes the TD error
    def critic_loss_function(critic_params):
        td_errors = calculate_td_error_fn(states = jax.lax.stop_gradient(states),
                                           actions = jax.lax.stop_gradient(actions),
                                           rewards = jax.lax.stop_gradient(rewards),
                                           next_states = jax.lax.stop_gradient(next_states),
                                           dones = jax.lax.stop_gradient(dones),
                                           temperature = jax.lax.stop_gradient(temperature),
                                           critic_params = critic_params,
                                           critic_target_params = jax.lax.stop_gradient(critic_target_params),
                                           next_log_policy = jax.lax.stop_gradient(next_log_policy))
        weighted_td_error_loss = jnp.mean(jax.lax.stop_gradient(weights) * td_errors)

        return weighted_td_error_loss
    grads = jax.grad(critic_loss_function)(critic_params)
    clipped_grads = clip_grads(grads, max_norm=critic_grad_max_norm)

    updates, critic_opt_state = critic_optimizer.update(
        clipped_grads,
        critic_opt_state,
        critic_params
    )

    critic_params = optax.apply_updates(critic_params, updates)

    critic_loss = critic_loss_function(critic_params)

    td_errors = calculate_td_error_fn(states = states,
                                      actions = actions,
                                      rewards = rewards,
                                      next_states = next_states,
                                      dones = dones,
                                      temperature = temperature,
                                      critic_params = critic_params,
                                      critic_target_params = critic_target_params,
                                      next_log_policy = next_log_policy)

    return critic_params, critic_opt_state, critic_loss, td_errors


def actor_update(actor_optimizer,
                 sample_actions_func,
                 states : jnp.ndarray,
                 temperature: float,
                 critic : nn.Module,
                 critic_params : jnp.ndarray,
                 actor_params : jnp.ndarray,
                 actor_opt_state : jnp.ndarray,
                 actor_grad_max_norm: float,
                 normal_distribution: jnp.ndarray,
                 actor: nn.Module,
                 std_min: float,
                 std_max: float,
                 print_bool: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Minimises entropy-regularised Q-values
    def actor_loss_function(actor_params):
        # Sample actions from the actor
        mean, std_0_1 = actor.apply(actor_params, jax.lax.stop_gradient(states))
        std = (std_max - std_min) * std_0_1 + std_min  # Fix to [0.01, 0.1]

        # Sample actions using normal distribution
        actions = jax.lax.stop_gradient(normal_distribution) * std + mean
        if actions.ndim == 0:
                    actions = jnp.expand_dims(actions, axis=0)
        # Turn states form (7,) to (1, 7)
        states_new = jnp.expand_dims(jax.lax.stop_gradient(states), axis=0)

        log_probs = gaussian_likelihood(actions, mean, std)

        # Entropy regualrised Q-values
        q1, q2 = critic.apply(jax.lax.stop_gradient(critic_params), jax.lax.stop_gradient(states_new), actions)
        q_min = jnp.minimum(q1, q2)

        entropy_loss = temperature * log_probs
        # Entropy regularised loss
        actor_loss = (entropy_loss - q_min).mean()
        # Clip actor loss
        return actor_loss
    def actor_loss_function_print(actor_params):
        # Sample actions from the actor
        actions, mean, std = sample_actions_func(states = jax.lax.stop_gradient(states),
                                                 actor_params = actor_params,
                                                 normal_dist = jax.lax.stop_gradient(normal_distribution),) # Module
        if actions.ndim == 0:
                    actions = jnp.expand_dims(actions, axis=0)
        log_probs = gaussian_likelihood(actions, mean, std)

        # Entropy regualrised Q-values
        q1, q2 = critic.apply(jax.lax.stop_gradient(critic_params), jax.lax.stop_gradient(states), actions)
        q_min = jnp.minimum(q1, q2)

        entropy_loss = temperature * log_probs

        # Entropy regularised loss
        actor_loss = (entropy_loss - q_min).mean()
        if print_bool:
            print(f'Actions: {actions}, Mean: {mean}, Std: {std}')
            print(f'Log probs: {log_probs}')
            print(f'Q1: {q1}')
            print(f'Q2: {q2}')
            print(f'Q_min: {q_min}')
            print(f'Entropy loss: {entropy_loss}')
        return actor_loss
    
    grads = jax.grad(actor_loss_function)(actor_params)
    clipped_grads = clip_grads(grads, max_norm=actor_grad_max_norm)
    if print_bool:
        print(f'grads: {grads}')

    updates, actor_opt_state = actor_optimizer.update(
        clipped_grads,
        actor_opt_state,
        actor_params
    )

    actor_params = optax.apply_updates(actor_params, updates)

    actor_loss = actor_loss_function_print(actor_params)

    if print_bool:
        print(f'Actor opt state: {actor_opt_state}')

    return actor_params, actor_opt_state, actor_loss


def temperature_update(temperature_optimizer,
                       temperature: float,
                       temperature_opt_state: jnp.ndarray,
                       log_probs: jnp.ndarray,
                       target_entropy: float,
                       temperature_grad_max_norm: float,
                       print_bool: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Aim to maintain entropy
    # Entropy is the average log probability of the actions
    def temperature_loss_fn(temperature):
        return -temperature * (jax.lax.stop_gradient(log_probs) + target_entropy).mean() # Rem. target entropy is constant

    grads = jax.grad(temperature_loss_fn)(temperature)
    clipped_grads = clip_grads(grads, max_norm=temperature_grad_max_norm)
    updates, temperature_opt_state = temperature_optimizer.update(clipped_grads, temperature_opt_state)
    temperature = optax.apply_updates(temperature, updates)

    temperature_loss = temperature_loss_fn(temperature)

    return temperature, temperature_opt_state, temperature_loss

def update_target_params(tau: float,
                         target_params: jnp.ndarray,
                         params: jnp.ndarray,
                         print_bool: bool) -> jnp.ndarray:
    target_params = jax.tree_util.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)
    return target_params

def update_sac(sample_actions_func,
               critic_update_func,
               actor_update_func,
               temperature_update_func,
               update_target_params_critic_func,
               # Training data
               states: jnp.ndarray,
               actions: jnp.ndarray,
               rewards: jnp.ndarray,
               next_states: jnp.ndarray,
               dones: jnp.ndarray,
               weights: jnp.ndarray,
               # Random Gaussian noise
               normal_dist: jnp.ndarray,
               critic_params: jnp.ndarray,
               critic_target_params: jnp.ndarray,
               critic_opt_state: jnp.ndarray,
               # Actor
               actor_params: jnp.ndarray,
               actor_opt_state: jnp.ndarray,
               # Temperature
               temperature: float,
               temperature_opt_state: jnp.ndarray,
               print_bool: bool
               ) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, \
                                            jnp.ndarray, jnp.ndarray, float, \
                                            float, jnp.ndarray, float, \
                                                jnp.ndarray]:

    # Perform SAC update step
    actions, mean, std = sample_actions_func(states = states,
                                             actor_params = actor_params,
                                             normal_dist = normal_dist)    
    
    log_probs = gaussian_likelihood(actions, mean, std)
    critic_params, critic_opt_state, critic_loss, td_errors = critic_update_func(states = states,
                                                                                 actions = actions,
                                                                                 rewards = rewards,
                                                                                 next_states = next_states,
                                                                                 dones = dones,
                                                                                 weights = weights,
                                                                                 critic_params = critic_params,
                                                                                 critic_opt_state = critic_opt_state,
                                                                                 critic_target_params = critic_target_params,
                                                                                 temperature = temperature,
                                                                                 next_log_policy = log_probs)  
    critic_loss = jax.lax.stop_gradient(critic_loss)
    td_errors = jax.lax.stop_gradient(td_errors)

    actor_params, actor_opt_state, actor_loss  = actor_update_func(states = states,
                                                                   temperature = temperature,
                                                                   critic_params = critic_params,
                                                                   actor_params = actor_params,
                                                                   actor_opt_state = actor_opt_state,
                                                                   normal_distribution = normal_dist)

    actor_loss = jax.lax.stop_gradient(actor_loss)

    temperature, temperature_opt_state, temperature_loss = temperature_update_func(temperature = temperature,
                                                                                   temperature_opt_state = temperature_opt_state,
                                                                                   log_probs = log_probs)
    
    temperature_loss = jax.lax.stop_gradient(temperature_loss)
    critic_target_params = update_target_params_critic_func(target_params = critic_target_params,
                                                            params = critic_params)
    

    

    # Have to update buffer still, but isn't jittable.
    if print_bool:
        print(f'Actions: {actions}')
        print(f'Mean: {mean}')
        print(f'Std: {std}')
        print(f'Log probs: {log_probs}')
        print(f'Rewards: {rewards}')
        print(f'Critic loss: {critic_loss}')
        print(f'TD errors: {td_errors}')
        print(f'Actor loss: {actor_loss}')
    
    return critic_params, critic_opt_state, critic_loss, td_errors, \
            actor_params, actor_opt_state, actor_loss, \
            temperature, temperature_opt_state, temperature_loss, \
            critic_target_params