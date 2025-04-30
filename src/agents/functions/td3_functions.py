import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import Callable, Tuple

@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads

def calculate_td_error(states: jnp.ndarray,
                      actions: jnp.ndarray,
                      rewards: jnp.ndarray,
                      next_states: jnp.ndarray,
                      dones: jnp.ndarray,
                      gamma: float,
                      critic_params: jnp.ndarray,
                      critic_target_params: jnp.ndarray,
                      critic: nn.Module,
                      next_actions: jnp.ndarray) -> jnp.ndarray:
    """Calculate TD error for TD3 using clipped double Q-learning."""
    q1, q2 = critic.apply(critic_params, states, actions)
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)  # Clipped double Q-learning
    td_target = rewards + gamma * (1 - dones) * next_q
    td_errors = 0.5 * ((td_target - q1)**2 + (td_target - q2)**2)
    return td_errors.astype(jnp.float32)  # Ensure float32 output

def huber_loss(td_error, delta):
    abs_error = jnp.abs(td_error)
    is_small = abs_error <= delta
    small_loss = 0.5 * td_error**2
    large_loss = delta * (abs_error - 0.5 * delta)
    return jnp.where(is_small, small_loss, large_loss)

def critic_update(critic_optimiser,
                 calculate_td_error_fcn: Callable,
                 critic_params: jnp.ndarray,
                 critic_opt_state: jnp.ndarray,
                 critic_grad_max_norm: float,
                 buffer_weights: jnp.ndarray,
                 states: jnp.ndarray,
                 actions: jnp.ndarray,
                 rewards: jnp.ndarray,
                 next_states: jnp.ndarray,
                 dones: jnp.ndarray,
                 critic_target_params: jnp.ndarray,
                 next_actions: jnp.ndarray,
                 delta: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update the critic networks."""
    def loss_fcn(params):
        td_errors = calculate_td_error_fcn(
            states=jax.lax.stop_gradient(states),
            actions=jax.lax.stop_gradient(actions),
            rewards=jax.lax.stop_gradient(rewards),
            next_states=jax.lax.stop_gradient(next_states),
            dones=jax.lax.stop_gradient(dones),
            critic_params=params,
            critic_target_params=jax.lax.stop_gradient(critic_target_params),
            next_actions=jax.lax.stop_gradient(next_actions)
        )
        loss_per_sample = huber_loss(td_errors, delta)
        weighted_loss = jnp.mean(jax.lax.stop_gradient(buffer_weights) * loss_per_sample)
        return weighted_loss.astype(jnp.float32), td_errors  # Ensure float32 output

    grads, _ = jax.grad(loss_fcn, has_aux=True)(critic_params)
    clipped_grads = clip_grads(grads, max_norm=critic_grad_max_norm)
    updates, critic_opt_state = critic_optimiser.update(clipped_grads, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    critic_loss, td_errors = loss_fcn(critic_params)
    return critic_params, critic_opt_state, critic_loss, td_errors

def actor_update(actor_optimiser,
                actor: nn.Module,
                critic: nn.Module,
                actor_grad_max_norm: float,
                states: jnp.ndarray,
                critic_params: jnp.ndarray,
                actor_params: jnp.ndarray,
                actor_opt_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update the actor network."""
    def loss_fcn(params):
        actions = actor.apply(params, jax.lax.stop_gradient(states))
        q1, _ = critic.apply(jax.lax.stop_gradient(critic_params), jax.lax.stop_gradient(states), actions)
        return -q1.mean().astype(jnp.float32)  # Ensure float32 output

    grads = jax.grad(loss_fcn)(actor_params)
    clipped_grads = clip_grads(grads, max_norm=actor_grad_max_norm)
    updates, actor_opt_state = actor_optimiser.update(clipped_grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    actor_loss = loss_fcn(actor_params)
    return actor_params, actor_opt_state, actor_loss

def update_td3(actor: nn.Module,
              actor_params: jnp.ndarray,
              actor_opt_state: jnp.ndarray,
              states: jnp.ndarray,
              actions: jnp.ndarray,
              rewards: jnp.ndarray,
              next_states: jnp.ndarray,
              dones: jnp.ndarray,
              buffer_weights: jnp.ndarray,
              critic_params: jnp.ndarray,
              critic_target_params: jnp.ndarray,
              critic_opt_state: jnp.ndarray,
              critic_update_lambda: Callable,
              actor_update_lambda: Callable,
              tau: float,
              policy_delay: int,
              clipped_noise: jnp.ndarray,
              step: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Main TD3 update function."""
    # 1. Sample next actions with noise
    next_actions = actor.apply(actor_params, next_states)
    next_actions = next_actions + clipped_noise

    # 2. Update the critic
    critic_params, critic_opt_state, critic_loss, td_errors = critic_update_lambda(
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        buffer_weights=buffer_weights,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        critic_target_params=critic_target_params,
        next_actions=next_actions
    )

    # 3. Delayed policy updates
    actor_params, actor_opt_state, actor_loss = jax.lax.cond(
        step % policy_delay == 0,
        lambda: actor_update_lambda(
            states=states,
            critic_params=critic_params,
            actor_params=actor_params,
            actor_opt_state=actor_opt_state
        ),
        lambda: (actor_params, actor_opt_state, jnp.array(0.0, dtype=jnp.float32))  # Ensure float32 type
    )

    # 4. Update target networks
    critic_target_params = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1.0 - tau) * tp,
        critic_params,
        critic_target_params
    )

    return critic_params, critic_opt_state, critic_loss, td_errors, \
           actor_params, actor_opt_state, actor_loss, critic_target_params

def critic_warm_up_update(actor : nn.Module,
                          actor_params: jnp.ndarray,
                          states: jnp.ndarray,
                          actions: jnp.ndarray,
                          rewards: jnp.ndarray,
                          next_states: jnp.ndarray,
                          dones: jnp.ndarray,
                          buffer_weights: jnp.ndarray,
                          critic_params: jnp.ndarray,
                          critic_target_params: jnp.ndarray,
                          critic_opt_state: jnp.ndarray,
                          critic_update_lambda: Callable,
                          clipped_noise: jnp.ndarray,
                          tau: float,
                          delta: float):
    # 1. Sample next actions with noise
    next_actions = actor.apply(actor_params, next_states)
    next_actions = next_actions + clipped_noise

    # 2. Update the critic
    critic_params, critic_opt_state, critic_loss, td_errors = critic_update_lambda(
                                            critic_params=critic_params,
                                            critic_opt_state=critic_opt_state,
                                            buffer_weights=buffer_weights,
                                            states=states,
                                            actions=actions,
                                            rewards=rewards,
                                            next_states=next_states,
                                            dones=dones,
                                            critic_target_params=critic_target_params,
                                            next_actions=next_actions)
    
    # 3. Update target networks
    critic_target_params = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1.0 - tau) * tp,
        critic_params,
        critic_target_params
    )

    return critic_params, critic_opt_state, critic_target_params, critic_loss

def lambda_compile_calculate_td_error(critic, gamma):
    return jax.jit(
        partial(calculate_td_error,
                critic=critic,
                gamma=gamma),
        static_argnames=['critic', 'gamma']
    )

def lambda_compile_td3(critic_optimiser,
                      critic: nn.Module,
                      critic_grad_max_norm: float,
                      actor_optimiser,
                      actor: nn.Module,
                      actor_grad_max_norm: float,
                      gamma: float,
                      tau: float,
                      policy_delay: int,
                      delta: float):
    """Compile TD3 functions with JIT."""
    calculate_td_error_lambda = jax.jit(
        partial(calculate_td_error,
                critic=critic,
                gamma=gamma),
        static_argnames=['critic', 'gamma']
    )

    critic_update_lambda = jax.jit(
        partial(critic_update,
                critic_optimiser=critic_optimiser,
                calculate_td_error_fcn=calculate_td_error_lambda,
                critic_grad_max_norm=critic_grad_max_norm,
                delta=delta),
        static_argnames=['critic_optimiser', 'calculate_td_error_fcn', 'critic_grad_max_norm', 'delta']
    )

    actor_update_lambda = jax.jit(
        partial(actor_update,
                actor_optimiser=actor_optimiser,
                actor=actor,
                critic=critic,
                actor_grad_max_norm=actor_grad_max_norm),
        static_argnames=['actor_optimiser', 'actor', 'critic', 'actor_grad_max_norm']
    )

    update_td3_lambda = jax.jit(
        partial(update_td3,
                actor=actor,
                critic_update_lambda=critic_update_lambda,
                actor_update_lambda=actor_update_lambda,
                tau=tau,
                policy_delay=policy_delay),
        static_argnames=['actor', 'critic_update_lambda', 'actor_update_lambda', 'tau', 'policy_delay']
    )

    critic_warm_up_update_lambda = jax.jit(
        partial(critic_warm_up_update,
                actor=actor,
                critic_update_lambda=critic_update_lambda,
                tau=tau),
        static_argnames=['actor', 'critic_update_lambda', 'tau']
    )
    return update_td3_lambda, calculate_td_error_lambda , critic_warm_up_update_lambda


