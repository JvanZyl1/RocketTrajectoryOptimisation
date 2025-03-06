import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from src.agents.functions.soft_actor_critic_functions import gaussian_likelihood

def update_worker_actor(states: jnp.ndarray,
                        actor_params_worker: jnp.ndarray,
                        actor_opt_state_worker: jnp.ndarray,
                        temperature_worker: float,
                        temperature_opt_state_worker: jnp.ndarray,
                        normal_distribution: jnp.ndarray,
                        central_critic_params: jnp.ndarray,
                        sample_actions_func: Callable,
                        actor_update_func: Callable,
                        temperature_update_func: Callable,
                        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, float]:
    # Worker: agent update: update actor and temperature
    actions, mean, std = sample_actions_func(states = states,
                                                 actor_params = actor_params_worker,
                                                 normal_dist = normal_distribution)
    
    log_probs = gaussian_likelihood(actions, mean, std)
    
    actor_params_worker, actor_opt_state_worker, actor_loss_worker = actor_update_func(states = states,
                                                                      temperature = temperature_worker,
                                                                      critic_params = central_critic_params,
                                                                      actor_params = actor_params_worker,
                                                                      actor_opt_state =  actor_opt_state_worker,
                                                                      normal_distribution = normal_distribution)
    
    actor_loss_worker = jax.lax.stop_gradient(actor_loss_worker)

    temperature_worker, temperature_opt_state_worker, temperature_loss_worker = temperature_update_func(temperature = temperature_worker,
                                                                                        temperature_opt_state = temperature_opt_state_worker,
                                                                                        log_probs = log_probs)
    temperature_loss_worker = jax.lax.stop_gradient(temperature_loss_worker)

    return (actor_params_worker, actor_opt_state_worker, actor_loss_worker,\
                temperature_worker, temperature_opt_state_worker, temperature_loss_worker)

def update_central_agent(sample_actions_func,
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
               central_critic_params: jnp.ndarray,
               central_critic_target_params: jnp.ndarray,
               central_critic_opt_state: jnp.ndarray,
               # Actor
               central_actor_params: jnp.ndarray,
               central_actor_opt_state: jnp.ndarray,
               # Temperature
               central_temperature: float,
               central_temperature_opt_state: jnp.ndarray,
               print_bool: bool
               ) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, \
                                            jnp.ndarray, jnp.ndarray, float, \
                                            float, jnp.ndarray, float, \
                                                jnp.ndarray]:

    # Perform SAC update step
    actions, mean, std = sample_actions_func(states = states,
                                             actor_params = central_actor_params,
                                             normal_dist = normal_dist)    
    
    log_probs = gaussian_likelihood(actions, mean, std)
    central_critic_params, central_critic_opt_state, central_critic_loss, td_errors = critic_update_func(states = states,
                                                                                 actions = actions,
                                                                                 rewards = rewards,
                                                                                 next_states = next_states,
                                                                                 dones = dones,
                                                                                 weights = weights,
                                                                                 critic_params = central_critic_params,
                                                                                 critic_opt_state = central_critic_opt_state,
                                                                                 critic_target_params = central_critic_target_params,
                                                                                 temperature = central_temperature,
                                                                                 next_log_policy = log_probs)  
    central_critic_loss = jax.lax.stop_gradient(central_critic_loss)
    td_errors = jax.lax.stop_gradient(td_errors)

    central_actor_params, central_actor_opt_state, central_actor_loss  = actor_update_func(states = states,
                                                                   temperature = central_temperature,
                                                                   critic_params = central_critic_params,
                                                                   actor_params = central_actor_params,
                                                                   actor_opt_state = central_actor_opt_state,
                                                                   normal_distribution = normal_dist)

    central_actor_loss = jax.lax.stop_gradient(central_actor_loss)

    central_temperature, central_temperature_opt_state, central_temperature_loss = temperature_update_func(temperature = central_temperature,
                                                                                   temperature_opt_state = central_temperature_opt_state,
                                                                                   log_probs = log_probs)
    
    central_temperature_loss = jax.lax.stop_gradient(central_temperature_loss)
    central_critic_target_params = update_target_params_critic_func(target_params = central_critic_target_params,
                                                            params = central_critic_params)

    # Have to update buffer still, but isn't jittable.
    if print_bool:
        print(f'Actions: {actions}')
        print(f'Mean: {mean}')
        print(f'Std: {std}')
        print(f'Log probs: {log_probs}')
        print(f'Rewards: {rewards}')
        print(f'Critic loss: {central_critic_loss}')
        print(f'TD errors: {td_errors}')
        print(f'Actor loss: {central_actor_loss}')
    
    return central_critic_params, central_critic_opt_state, central_critic_loss, td_errors, \
            central_actor_params, central_actor_opt_state, central_actor_loss, \
            central_temperature, central_temperature_opt_state, central_temperature_loss, \
            central_critic_target_params

def update_marl(update_worker_actor_func, update_central_agent_func,
                all_workers_actor_params, all_workers_actor_opt_state, all_workers_temperatures, all_workers_temperature_opt_state,
                states, actions, rewards, next_states, dones, weights, normal_distributions,
                central_critic_params, central_critic_target_params, central_critic_opt_state,
                central_actor_params, central_actor_opt_state, central_temperature, central_temperature_opt_state):
    """
    Update all workers and central agent
    """
    number_of_workers = len(all_workers_actor_params)
    
    # Initialize lists to store updated parameters and losses
    updated_worker_params = []
    updated_worker_opt_states = []
    updated_worker_temperatures = []
    updated_worker_temp_opt_states = []
    actor_loss_workers = []
    temperature_loss_workers = []

    # Update each worker
    for i in range(number_of_workers):
        worker_actor_params = all_workers_actor_params[i]
        worker_actor_opt_state = all_workers_actor_opt_state[i]
        worker_temperature = all_workers_temperatures[i]
        worker_temperature_opt_state = all_workers_temperature_opt_state[i]

        # Update worker
        new_actor_params, new_actor_opt_state, actor_loss, \
        new_temperature, new_temp_opt_state, temp_loss = \
            update_worker_actor_func(
                states=states[i],
                actor_params_worker=worker_actor_params,
                actor_opt_state_worker=worker_actor_opt_state,
                temperature_worker=worker_temperature,
                temperature_opt_state_worker=worker_temperature_opt_state,
                normal_distribution=normal_distributions[i],
                central_critic_params=central_critic_params
            )
        
        # Store updated values
        updated_worker_params.append(new_actor_params)
        updated_worker_opt_states.append(new_actor_opt_state)
        updated_worker_temperatures.append(new_temperature)
        updated_worker_temp_opt_states.append(new_temp_opt_state)
        actor_loss_workers.append(actor_loss)
        temperature_loss_workers.append(temp_loss)

    # Update central agent using keyword arguments
    central_critic_params, central_critic_opt_state, central_critic_loss, td_errors, \
    central_actor_params, central_actor_opt_state, central_actor_loss, \
    central_temperature, central_temperature_opt_state, central_temperature_loss, \
    central_critic_target_params = update_central_agent_func(
        states=states[-1], 
        actions=actions[-1], 
        rewards=rewards[-1], 
        next_states=next_states[-1], 
        dones=dones[-1], 
        weights=weights[-1],
        normal_dist=normal_distributions[-1], 
        central_critic_params=central_critic_params, 
        central_critic_target_params=central_critic_target_params,
        central_critic_opt_state=central_critic_opt_state,
        central_actor_params=central_actor_params,
        central_actor_opt_state=central_actor_opt_state,
        central_temperature=central_temperature,
        central_temperature_opt_state=central_temperature_opt_state
    )
    
    # Convert lists to JAX arrays where needed
    actor_loss_workers = jnp.array(actor_loss_workers)
    temperature_loss_workers = jnp.array(temperature_loss_workers)

    return (central_critic_params, central_critic_opt_state, central_critic_loss, td_errors,
            central_actor_params, central_actor_opt_state, central_actor_loss,
            central_temperature, central_temperature_opt_state, central_temperature_loss,
            central_critic_target_params, 
            updated_worker_params, updated_worker_opt_states,
            updated_worker_temperatures, updated_worker_temp_opt_states,
            actor_loss_workers, temperature_loss_workers)
