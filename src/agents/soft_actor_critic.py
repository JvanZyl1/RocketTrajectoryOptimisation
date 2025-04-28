import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.soft_actor_critic_functions import lambda_compile_sac, gaussian_likelihood
from src.agents.functions.plotter import agent_plotter_sac
from src.agents.functions.buffers import PERBuffer
from src.agents.functions.networks import Actor, DoubleCritic
    
class SoftActorCritic:
    def __init__(self,
                 state_dim : int,
                 action_dim : int,
                 flight_phase : str,
                 # Dimensions
                 hidden_dim_actor : int,
                 number_of_hidden_layers_actor : int,
                 hidden_dim_critic : int,
                 number_of_hidden_layers_critic : int,
                 # Hyper-parameters
                 temperature_initial : float,
                 gamma : float,
                 tau : float,
                 alpha_buffer : float,
                 beta_buffer : float,
                 beta_decay_buffer : float,
                 buffer_size : int,
                 trajectory_length : int,
                 batch_size : int,
                 # Learning rates
                 critic_learning_rate : float,
                 actor_learning_rate : float,
                 temperature_learning_rate : float,
                 # Grad max norms
                 critic_grad_max_norm : float,
                 actor_grad_max_norm : float,
                 temperature_grad_max_norm : float,
                 # Max std
                 max_std : float):
        
        self.rng_key = jax.random.PRNGKey(0)
        
        self.save_path = f'results/VanillaSAC/{flight_phase}/'
        self.flight_phase = flight_phase
        self.buffer = PERBuffer(
            gamma=gamma,
            alpha=alpha_buffer,
            beta=beta_buffer,
            beta_decay=beta_decay_buffer,
            buffer_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            trajectory_length=trajectory_length,
            batch_size=batch_size
        )
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f'data/agent_saves/VanillaSAC/{flight_phase}/runs/{self.run_id}')

        self.max_std = max_std

        self.actor = Actor(action_dim=action_dim,
                           hidden_dim=hidden_dim_actor,
                           number_of_hidden_layers=number_of_hidden_layers_actor)
        self.actor_params = self.actor.init(self.get_subkey(), jnp.zeros((1, state_dim)))
        self.actor_opt_state = optax.adam(learning_rate=actor_learning_rate).init(self.actor_params)

        self.critic = DoubleCritic(state_dim=state_dim,
                                   action_dim=action_dim,
                                   hidden_dim=hidden_dim_critic,
                                   number_of_hidden_layers=number_of_hidden_layers_critic)
        self.critic_params = self.critic.init(self.get_subkey(), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic_opt_state = optax.adam(learning_rate=critic_learning_rate).init(self.critic_params)
        self.critic_target_params = self.critic_params

        self.temperature = temperature_initial
        self.temperature_opt_state = optax.adam(learning_rate=temperature_learning_rate).init(self.temperature)

        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm

        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_max_norm = actor_grad_max_norm

        self.temperature_learning_rate = temperature_learning_rate
        self.temperature_grad_max_norm = temperature_grad_max_norm

        self.gamma = gamma
        self.tau = tau

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_dim_actor = hidden_dim_actor
        self.number_of_hidden_layers_actor = number_of_hidden_layers_actor
        self.hidden_dim_critic = hidden_dim_critic
        self.number_of_hidden_layers_critic = number_of_hidden_layers_critic
        self.temperature_initial = temperature_initial
        self.alpha_buffer = alpha_buffer
        self.beta_buffer = beta_buffer
        self.beta_decay_buffer = beta_decay_buffer
        self.buffer_size = buffer_size
        self.trajectory_length = trajectory_length
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.temperature_learning_rate = temperature_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm
        self.actor_grad_max_norm = actor_grad_max_norm
        self.temperature_grad_max_norm = temperature_grad_max_norm
        self.batch_size = batch_size

        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda \
            = lambda_compile_sac(critic_optimiser = optax.adam(learning_rate = self.critic_learning_rate),
                                 critic = self.critic,
                                 critic_grad_max_norm = self.critic_grad_max_norm,
                                 actor_optimiser = optax.adam(learning_rate = self.actor_learning_rate),
                                 actor = self.actor,
                                 actor_grad_max_norm = self.actor_grad_max_norm,
                                 temperature_optimiser = optax.adam(learning_rate=self.temperature_learning_rate),
                                 temperature_grad_max_norm = self.temperature_grad_max_norm,
                                 gamma = self.gamma,
                                 tau = self.tau,
                                 target_entropy = -self.action_dim,
                                 initial_temperature = self.temperature_initial)

        # LOGGING
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.critic_warm_up_step_idx = 0

    def reset(self):
        # LOGGING
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.rng_key = jax.random.PRNGKey(0)
        self.buffer.reset()
        self.critic_warm_up_step_idx = 0

    def get_subkey(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey
    
    def get_normal_distributions_batched(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.batch_size, self.action_dim))) * self.max_std
        return normal_distribution
    
    def get_normal_distribution(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.action_dim, ))) * self.max_std
        return normal_distribution
    
    def critic_warm_up_step(self):
        states, actions, rewards, next_states, dones, _, _ = self.buffer(self.get_subkey())
        self.critic_params, self.critic_opt_state, self.critic_target_params, critic_loss_warm_up = self.critic_warm_up_update_lambda(actor_params = self.actor_params,
                                                                                                                                      normal_distribution_for_next_actions = self.get_normal_distributions_batched(),
                                                                                                                                      states = states,
                                                                                                                                      actions = actions,
                                                                                                                                      rewards = rewards,
                                                                                                                                      next_states = next_states,
                                                                                                                                      dones = dones,
                                                                                                                                      critic_params = self.critic_params,
                                                                                                                                      critic_target_params = self.critic_target_params,
                                                                                                                                      critic_opt_state = self.critic_opt_state)
        self.writer.add_scalar('CriticWarmUp/Loss', np.array(critic_loss_warm_up), self.critic_warm_up_step_idx)
        self.critic_warm_up_step_idx += 1
        return critic_loss_warm_up

    def calculate_td_error(self,
                           state: jnp.ndarray,
                           action: jnp.ndarray,
                           reward: jnp.ndarray,
                           next_state: jnp.ndarray,
                           done: jnp.ndarray) -> jnp.ndarray:
        # This is non-batched.
        next_action_mean, next_action_std = self.actor.apply(self.actor_params, next_state)
        next_action = self.get_normal_distribution() * next_action_std + next_action_mean
        next_log_policy = gaussian_likelihood(next_action, next_action_mean, next_action_std)
        td_errors =  self.calculate_td_error_lambda(states = state,
                                                    actions = action,
                                                    rewards = reward,
                                                    next_states = next_state,
                                                    dones = done,
                                                    temperature = self.temperature,
                                                    critic_params = self.critic_params,
                                                    critic_target_params = self.critic_target_params,
                                                    next_actions = next_action,
                                                    next_log_policy = next_log_policy)
        return td_errors

    def calculate_td_error_vmap(self,
                               states: jnp.ndarray,
                               actions: jnp.ndarray,
                               rewards: jnp.ndarray,
                               next_states: jnp.ndarray,
                               dones: jnp.ndarray) -> jnp.ndarray:
        """Vectorized version of calculate_td_error using vmap."""
        # Ensure proper shapes
        states = jnp.reshape(states, (len(states), -1))  # (batch_size, state_dim)
        actions = jnp.reshape(actions, (len(actions), -1))  # (batch_size, action_dim)
        rewards = jnp.reshape(rewards, (len(rewards), 1))  # (batch_size, 1)
        next_states = jnp.reshape(next_states, (len(next_states), -1))  # (batch_size, state_dim)
        dones = jnp.reshape(dones, (len(dones), 1))  # (batch_size, 1)
        
        # use vmap to get next_action_means, next_action_stds
        next_action_means, next_action_stds = jax.vmap(self.actor.apply, in_axes=(None, 0))(self.actor_params, next_states)
        
        # get normal distributions in batch with proper key splitting
        self.rng_key, *subkeys = jax.random.split(self.rng_key, len(next_states) + 1)
        normal_distributions = jax.vmap(lambda key: jax.random.normal(key, (self.action_dim,)) * self.max_std)(jnp.array(subkeys))
        
        # use vmap for next actions and log policies
        next_actions = jax.vmap(lambda mean, std, normal: normal * std + mean)(
            next_action_means, next_action_stds, normal_distributions
        )
        next_log_policies = jax.vmap(gaussian_likelihood)(next_actions, next_action_means, next_action_stds)
        next_log_policies = jnp.reshape(next_log_policies, (len(next_log_policies), 1))  # Ensure (batch_size, 1) shape

        def td_error_calc(state, action, reward, next_state, done, next_action, next_log_policy):
            return self.calculate_td_error_lambda(
                states=state,
                actions=action,
                rewards=reward,
                next_states=next_state,
                dones=done,
                temperature=self.temperature,
                critic_params=self.critic_params,
                critic_target_params=self.critic_target_params,
                next_actions=next_action,
                next_log_policy=next_log_policy
            )
        
        # Calculate TD errors for all transitions using vmap
        td_errors = jax.vmap(td_error_calc)(states, actions, rewards, next_states, dones, next_actions, next_log_policies)
        
        return td_errors

    def select_actions(self,
                       state : jnp.ndarray) -> jnp.ndarray:
        # This is non-batched.
        action_mean, action_std = self.actor.apply(self.actor_params, state)
        actions = self.get_normal_distribution() * jnp.squeeze(action_std) + jnp.squeeze(action_mean)
        return actions

    def select_actions_no_stochastic(self,
                                     state : jnp.ndarray) -> jnp.ndarray:
        # This is non-batched.
        action_mean, _ = self.actor.apply(self.actor_params, state)
        return action_mean

    def update_episode(self):
        self.critic_losses.append(self.critic_loss_episode)
        self.actor_losses.append(self.actor_loss_episode)
        self.temperature_losses.append(self.temperature_loss_episode)
        self.td_errors.append(self.td_errors_episode)
        mean_temperature = jnp.mean(jnp.asarray([jax.nn.softplus(t) for t in self.temperature_values_all_episode]))
        self.temperature_values.append(mean_temperature)
        self.number_of_steps.append(self.number_of_steps_episode)

        # Log episode metrics to TensorBoard
        self.writer.add_scalar('Episode/CriticLoss', np.array(self.critic_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorLoss', np.array(self.actor_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/TemperatureLoss', np.array(self.temperature_loss_episode), self.episode_idx)
        self.writer.add_histogram('Episode/TDError', np.array(self.td_errors_episode), self.episode_idx)
        self.writer.add_scalar('Episode/MeanTemperature', np.array(mean_temperature), self.episode_idx)
        self.writer.add_scalar('Episode/Temperature', np.array(jax.nn.softplus(self.temperature)), self.episode_idx)
        self.writer.add_scalar('Episode/NumberOfSteps', np.array(self.number_of_steps_episode), self.episode_idx)
        for layer_name, layer_params in self.actor_params['params'].items():
            for param_name, param in layer_params.items():
                self.writer.add_histogram(f'Episode/Actor/{layer_name}/{param_name}', np.array(param).flatten(), self.episode_idx)

        for layer_name, layer_params in self.critic_params['params'].items():
            for param_name, param in layer_params.items():
                self.writer.add_histogram(f'Episode/Critic/{layer_name}/{param_name}', np.array(param).flatten(), self.episode_idx)
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx += 1

    def update(self):
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(self.get_subkey())

        self.critic_params, self.critic_opt_state, critic_loss, td_errors, \
            self.actor_params, self.actor_opt_state, actor_loss, \
            self.temperature, self.temperature_opt_state, temperature_loss, \
            self.critic_target_params = self.update_function(actor_params = self.actor_params,
                                                             actor_opt_state = self.actor_opt_state,
                                                             normal_distribution_for_next_actions = self.get_normal_distributions_batched(),
                                                             normal_distribution_for_actions = self.get_normal_distributions_batched(),
                                                             states = states,
                                                             actions = actions,
                                                             rewards = rewards,
                                                             next_states = next_states,
                                                             dones = dones,
                                                             buffer_weights = weights_buffer,
                                                             temperature = self.temperature,
                                                             temperature_opt_state = self.temperature_opt_state,
                                                             critic_params = self.critic_params,
                                                             critic_target_params = self.critic_target_params,
                                                             critic_opt_state = self.critic_opt_state)
        self.buffer.update_priorities(index, td_errors)

        self.critic_loss_episode += critic_loss
        self.actor_loss_episode += actor_loss
        self.temperature_loss_episode += temperature_loss
        self.td_errors_episode += td_errors
        self.temperature_values_all_episode.append(self.temperature)
        self.number_of_steps_episode += 1
        self.step_idx += 1
        
        # Convert JAX arrays to NumPy arrays before logging
        critic_loss_np = np.array(critic_loss)
        actor_loss_np = np.array(actor_loss)
        temperature_loss_np = np.array(temperature_loss)
        td_errors_np = np.array(td_errors)
        temperature_np = np.array(jax.nn.softplus(self.temperature))
        
        self.writer.add_scalar('Steps/CriticLoss', critic_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActorLoss', actor_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/TemperatureLoss', temperature_loss_np, self.step_idx)
        
        # Log TD errors as scalar instead of histogram
        self.writer.add_scalar('Steps/TDError/mean', np.mean(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/std', np.std(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/max', np.max(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/min', np.min(td_errors_np), self.step_idx)
        
        self.writer.add_scalar('Steps/Temperature', temperature_np, self.step_idx)
        self.writer.add_scalar('Steps/NumberOfSteps', self.number_of_steps_episode, self.step_idx)

        # Helper function to safely log parameter histograms
        '''
        def safe_log_histogram(tag, values, step):
            values_np = np.array(values).flatten()
            if len(values_np) > 0 and not np.all(values_np == values_np[0]):
                # Only log if there's variation in the values
                self.writer.add_histogram(tag, values_np, step)
            else:
                # Log as scalar if all values are the same
                self.writer.add_scalar(f"{tag}/value", values_np[0] if len(values_np) > 0 else 0.0, step)

        for layer_name, layer_params in self.actor_params['params'].items():
            for param_name, param in layer_params.items():
                safe_log_histogram(f'Actor/{layer_name}/{param_name}', param, self.step_idx)

        for layer_name, layer_params in self.critic_params['params'].items():
            for param_name, param in layer_params.items():
                safe_log_histogram(f'Critic/{layer_name}/{param_name}', param, self.step_idx)
        '''

        self.writer.add_scalar('Steps/Temperature', np.array(jax.nn.softplus(self.temperature)), self.step_idx)

    def plotter(self):
        agent_plotter_sac(self)

    def save(self):
        file_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/soft-actor-critic.pkl'
        agent_state = {
            'inputs' : {
                'flight_phase' : self.flight_phase,
                'hidden_dim_actor' : self.hidden_dim_actor,
                'number_of_hidden_layers_actor' : self.number_of_hidden_layers_actor,
                'hidden_dim_critic' : self.hidden_dim_critic,
                'number_of_hidden_layers_critic' : self.number_of_hidden_layers_critic,
                'temperature_initial' : self.temperature_initial,
                'gamma' : self.gamma,
                'tau' : self.tau,
                'alpha_buffer' : self.alpha_buffer,
                'beta_buffer' : self.beta_buffer,
                'beta_decay_buffer' : self.beta_decay_buffer,
                'buffer_size' : self.buffer_size,
                'state_dim' : self.state_dim,
                'action_dim' : self.action_dim,
                'trajectory_length' : self.trajectory_length,
                'batch_size' : self.batch_size,
                'critic_learning_rate' : self.critic_learning_rate,
                'actor_learning_rate' : self.actor_learning_rate,
                'temperature_learning_rate' : self.temperature_learning_rate,
                'critic_grad_max_norm' : self.critic_grad_max_norm,
                'actor_grad_max_norm' : self.actor_grad_max_norm,
                'temperature_grad_max_norm' : self.temperature_grad_max_norm,
                'max_std' : self.max_std
            },
            'misc' : {
                'rng_key' : self.rng_key,
                'run_id' : self.run_id
            },
            'logging' : {
                'critic_loss_episode' : self.critic_loss_episode,
                'actor_loss_episode' : self.actor_loss_episode,
                'temperature_loss_episode' : self.temperature_loss_episode,
                'td_errors_episode' : self.td_errors_episode,
                'temperature_values_all_episode' : self.temperature_values_all_episode,
                'number_of_steps_episode' : self.number_of_steps_episode,
                'episode_idx' : self.episode_idx,
                'step_idx' : self.step_idx,
                'critic_losses' : self.critic_losses,
                'actor_losses' : self.actor_losses,
                'temperature_losses' : self.temperature_losses,
                'td_errors' : self.td_errors,
                'temperature_values' : self.temperature_values,
                'number_of_steps' : self.number_of_steps
            },
            'update' : {
                'critic_params' : self.critic_params,
                'critic_opt_state' : self.critic_opt_state,
                'critic_target_params' : self.critic_target_params,
                'actor_params' : self.actor_params,
                'actor_opt_state' : self.actor_opt_state,
                'temperature' : self.temperature,
                'temperature_opt_state' : self.temperature_opt_state,
                'buffer' : self.buffer
            }
        }

        with open(file_path, 'wb') as f:
                pickle.dump(agent_state, f)