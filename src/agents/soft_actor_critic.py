from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.networks import Actor
from src.agents.functions.networks import DoubleCritic as Critic
from src.agents.functions.buffers import PERInference as Buffer

from src.agents.functions.plotter import agent_plotter_sac as agent_plotter
from src.agents.functions.soft_actor_critic_functions import (sample_actions, calculate_td_error,
                                                              critic_update, actor_update,
                                                              temperature_update, update_target_params,
                                                              update_sac, gaussian_likelihood)

# Inference class
class SoftActorCritic:
    def __init__(self,
                 seed: int,
                 # Dimensions
                 state_dim: int,
                 action_dim: int,
                 hidden_dim_actor: int = 256,
                 number_of_hidden_layers_actor: int = 3,
                 hidden_dim_critic: int = 256,
                 std_min: float = 0.01,
                 std_max: float = 0.1,
                 # Hyperparameters
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 temperature_initial: float = 0.1,
                 critic_grad_max_norm: float = 1.0,
                 critic_lr: float = 3e-4,
                 critic_weight_decay: float = 0.0,
                 actor_grad_max_norm: float = 1.0,
                 actor_lr: float = 3e-3,
                 actor_weight_decay: float = 0.0,
                 temperature_lr: float = 5e-4,
                 temperature_grad_max_norm: float = 1.0,
                 # Buffer hyperparameters
                 alpha_buffer: float = 0.6,
                 beta_buffer: float = 0.4,
                 beta_decay_buffer: float = 0.99,
                 trajectory_length: int = 1,
                 buffer_size: int = 1000,
                 batch_size: int = 128,
                 model_name: str = "VerticalRising-SAC",
                 print_bool: bool = False):
        
        # Initialisation
        self.temperature_initial = temperature_initial
        self.temperature = temperature_initial
        self.target_entropy = -action_dim

        self.seed = seed
        self.rng_key = jax.random.PRNGKey(seed)

        self.save_path = f'results/{model_name}/'
        self.model_name = model_name

        # Buffer
        self.trajectory_length = trajectory_length
        self.buffer = Buffer(
            gamma=gamma,
            alpha=alpha_buffer,
            beta=beta_buffer,
            beta_decay=beta_decay_buffer,
            buffer_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            trajectory_length=trajectory_length, # Nstep
            batch_size=batch_size
        )

        # Batch size
        self.batch_size = batch_size

        # Networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim_actor = hidden_dim_actor
        self.hidden_dim_critic = hidden_dim_critic
        self.actor = Actor(action_dim=action_dim,
                           hidden_dim=hidden_dim_actor,
                           number_of_hidden_layers=number_of_hidden_layers_actor)
        self.std_min = std_min
        self.std_max = std_max
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.actor_params = self.actor.init(subkey, jnp.zeros((1, state_dim)))
        self.critic = Critic(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dim=hidden_dim_critic)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.critic_params = self.critic.init(subkey, jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic_target_params = self.critic_params

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau

        # Optimisers
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay
        self.critic_grad_max_norm = critic_grad_max_norm
        self.critic_optimizer = optax.adamaxw(learning_rate=self.critic_lr,
                                              weight_decay=self.critic_weight_decay)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        self.actor_lr = actor_lr
        self.actor_weight_decay = actor_weight_decay
        self.actor_grad_max_norm = actor_grad_max_norm
        self.actor_optimizer = optax.adamaxw(learning_rate=self.actor_lr,
                                                weight_decay=self.actor_weight_decay)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        self.temperature_lr = temperature_lr
        self.temperature_grad_max_norm = temperature_grad_max_norm
        self.temperature_optimizer = optax.adam(learning_rate=self.temperature_lr)
        self.temperature_opt_state = self.temperature_optimizer.init(self.temperature)
        
        # Logging
        self.critic_losses = []
        self.actor_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0

        # Print bool
        self.print_bool = print_bool

        # Create functions
        self.create_funcs()

        # Initialize TensorBoard writer
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer_dir = f'data/agent_saves/{model_name}/runs/{self.run_id}'
        self.writer = SummaryWriter(log_dir=self.writer_dir)
        self.step_idx = 0
        self.episode_idx = 0

    def create_funcs(self):
        if not self.print_bool:
            self.sample_actions_func = jax.jit(
                partial(sample_actions,
                        actor=self.actor,
                        std_min=self.std_min,
                        std_max=self.std_max,
                        print_bool=self.print_bool),
                static_argnames=['actor', 'std_min', 'std_max', 'print_bool']
            )

            self.calculate_td_error_func = jax.jit(
                partial(calculate_td_error,
                        critic=self.critic,
                        gamma=self.gamma,
                        print_bool=self.print_bool),
                static_argnames=['critic', 'gamma', 'print_bool']
            )

            self.critic_update_func = jax.jit(
                partial(critic_update,
                        critic_optimizer=self.critic_optimizer,
                        calculate_td_error_fn=self.calculate_td_error_func,
                        critic_grad_max_norm=self.critic_grad_max_norm,
                        print_bool=self.print_bool),
                static_argnames=['critic_optimizer', 'calculate_td_error_fn', 'critic_grad_max_norm', 'print_bool']
            )

            self.actor_update_func = jax.jit(
                partial(actor_update,
                        actor_optimizer=self.actor_optimizer,
                        sample_actions_func=self.sample_actions_func,
                        critic = self.critic,
                        actor_grad_max_norm=self.actor_grad_max_norm,
                        actor=self.actor,
                        std_min=self.std_min,
                        std_max=self.std_max,
                        print_bool=self.print_bool),
                static_argnames=['actor_optimizer', 'sample_actions_func', 'critic', 'actor_grad_max_norm', 'actor', 'std_min', 'std_max', 'print_bool']
            )

            self.temperature_update_func = jax.jit(
                partial(temperature_update,
                        temperature_optimizer=self.temperature_optimizer,
                        target_entropy=self.target_entropy,
                        temperature_grad_max_norm=self.temperature_grad_max_norm,
                        print_bool=self.print_bool),
                static_argnames=['temperature_optimizer', 'target_entropy', 'temperature_grad_max_norm', 'print_bool']
            )

            self.update_target_params_func = jax.jit(
                partial(update_target_params,
                        tau=self.tau,
                        print_bool=self.print_bool),
                static_argnames=['tau', 'print_bool']
            )

            self.update_sac_func = jax.jit(
                partial(update_sac,
                        sample_actions_func=self.sample_actions_func,
                        critic_update_func=self.critic_update_func,
                        actor_update_func=self.actor_update_func,
                        temperature_update_func=self.temperature_update_func,
                        update_target_params_critic_func=self.update_target_params_func,
                        print_bool=self.print_bool),
                static_argnames=['sample_actions_func', 'critic_update_func', 'actor_update_func', 'temperature_update_func', 'update_target_params_critic_func', 'print_bool']
            )

        else: # None jitable functions
            self.sample_actions_func = lambda states, actor_params, normal_dist: sample_actions(states, actor_params, normal_dist, self.actor, self.std_min, self.std_max, self.print_bool)
            self.calculate_td_error_func = lambda states, actions, rewards, next_states, dones, temperature, \
                                                    critic_params, critic_target_params, next_log_policy: calculate_td_error(states, actions, rewards, next_states, dones, temperature, self.gamma, critic_params, critic_target_params, self.critic, next_log_policy, self.print_bool)
            self.critic_update_func = lambda states, actions, rewards, next_states, dones, weights, critic_params, \
                                                    critic_opt_state, critic_target_params, temperature, next_log_policy: critic_update(self.critic_optimizer, self.calculate_td_error_func, states, actions, rewards, next_states, dones, weights, critic_params, critic_opt_state, \
                                                                                                                                            self.critic_grad_max_norm, critic_target_params, temperature, next_log_policy, \
                                                                                                                                                self.print_bool)
            self.actor_update_func = lambda states, temperature, critic_params, actor_params, actor_opt_state, normal_distribution: actor_update(self.actor_optimizer, self.sample_actions_func, states, temperature, self.critic, critic_params, actor_params, \
                                                                                                                                                           actor_opt_state, self.actor_grad_max_norm, normal_distribution, self.actor, self.std_min, self.std_max, self.print_bool)
            self.temperature_update_func = lambda temperature, temperature_opt_state, log_probs: temperature_update(self.temperature_optimizer, temperature, temperature_opt_state, log_probs \
                                                                                                                                , self.target_entropy, self.temperature_grad_max_norm, self.print_bool)
            self.update_target_params_func = lambda target_params, params: update_target_params(self.tau, target_params, params, self.print_bool)
            self.update_sac_func = lambda states, actions, rewards, next_states, dones, weights, normal_dist, critic_params, critic_target_params, critic_opt_state, \
                                             actor_params, actor_opt_state, temperature, temperature_opt_state: update_sac(self.sample_actions_func, self.critic_update_func, self.actor_update_func, \
                                                                                                                           self.temperature_update_func, self.update_target_params_func, states, actions, rewards, next_states, dones, weights, normal_dist, critic_params, 
                                                                                                                           critic_target_params, critic_opt_state, actor_params, actor_opt_state, temperature, temperature_opt_state, self.print_bool)               

    def reset(self, rng_key: jnp.ndarray) -> None:
        self.rng_key = rng_key
        self.critic_losses = []
        self.actor_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.buffer.reset(rng_key)
        self.create_funcs()
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.step_idx = 0
        self.episode_idx = 0

    def select_actions_no_stochatic(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Select deterministic actions for testing (no exploration noise).
        """
        mean, std_0_1 = self.actor.apply(self.actor_params, state)
        return mean

    def select_actions(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Select an action from the policy.
        
        Args:
            state: Current state (batch_size, state_dim).
        
        Returns:
            Action (batch_size, action_dim).
        """
        batch_size = state.shape[0]
        mean_shape = (batch_size, self.action_dim)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        normal_distribution = jax.random.normal(subkey, mean_shape)
        state = jnp.asarray(state)
        normal_distribution = jnp.asarray(normal_distribution)
        
        actions, _, _ = self.sample_actions_func(states=state,
                                                 actor_params=self.actor_params,
                                                 normal_dist=normal_distribution)
        return actions
    
    def log_probability_next_action(self, state: jnp.ndarray) -> jnp.ndarray:
        batch_size = state.shape[0]
        mean_shape = (batch_size, self.action_dim)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        normal_distribution = jax.random.normal(subkey, mean_shape)
        states = jnp.asarray(state)
        normal_distribution = jnp.asarray(normal_distribution)
        actions, mean, std = self.sample_actions_func(states=states,
                                                    actor_params=self.actor_params,
                                                    normal_dist=normal_distribution)
        
        log_probability = gaussian_likelihood(actions, mean, std)
        return log_probability
    
    def calculate_td_error(self,
                            states: jnp.ndarray,
                            actions: jnp.ndarray,
                            rewards: jnp.ndarray,
                            next_states: jnp.ndarray,
                            dones: jnp.ndarray) -> jnp.ndarray:
        
        next_log_policy = self.log_probability_next_action(next_states)
        td_errors = self.calculate_td_error_func(states = states,
                                                 actions = actions,
                                                 rewards = rewards,
                                                 next_states = next_states,
                                                 dones = dones,
                                                 temperature = self.temperature,
                                                 critic_params = self.critic_params,
                                                 critic_target_params = self.critic_target_params,
                                                 next_log_policy = next_log_policy)
        return td_errors
    
    def critic_update(self,
                      states: jnp.ndarray,
                      actions: jnp.ndarray,
                      rewards: jnp.ndarray,
                      next_states: jnp.ndarray,
                      dones: jnp.ndarray,
                      weights: jnp.ndarray) -> None:
        self.critic_params, self.critic_opt_state, critic_loss, td_errors = self.critic_update_func(states = states,
                                                                                                    actions = actions,
                                                                                                    rewards = rewards,
                                                                                                    next_states = next_states,
                                                                                                    dones = dones,
                                                                                                    weights = weights,
                                                                                                    critic_params = self.critic_params,
                                                                                                    critic_opt_state = self.critic_opt_state,
                                                                                                    critic_target_params = self.critic_target_params,
                                                                                                    temperature = self.temperature,
                                                                                                    next_log_policy = self.log_probability_next_action(next_states))
        
        self.critic_losses.append(critic_loss)
        self.td_errors = td_errors

    def actor_update(self,
                     states: jnp.ndarray,
                     actions: jnp.ndarray) -> None:
        # Generate random noise for the actions
        batch_size = states.shape[0]
        mean_shape = (batch_size, self.action_dim)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        normal_distribution = jax.random.normal(subkey, mean_shape)
        
        if self.print_bool:
            print(f"Training actor on states: {states}")
            print(f"Current actor params: {self.actor_params}")
        
        self.actor_params, self.actor_opt_state, actor_loss = self.actor_update_func(
            states=states,
            actions=actions,
            temperature=self.temperature,
            critic_params=self.critic_params,
            actor_params=self.actor_params,
            actor_opt_state=self.actor_opt_state,
            normal_distribution=normal_distribution  # Use the newly generated normal_distribution
        )

        
        if self.print_bool:
            print(f"Updated actor params: {self.actor_params}")
            print(f"Actor loss: {actor_loss}")
        
        self.actor_losses.append(actor_loss)

    def temperature_update(self,
                           states: jnp.ndarray) -> None:
        log_probs = self.log_probability_next_action(state = states)
        self.temperature, self.temperature_opt_state, temperature_loss = self.temperature_update_func(temperature = self.temperature,
                                                                                                      temperature_opt_state = self.temperature_opt_state,
                                                                                                      log_probs = log_probs)
        self.temperature_losses.append(temperature_loss)

    def target_update(self) -> None:
        self.critic_target_params = self.update_target_params_func(critic_target_params = self.critic_target_params,
                                                                   critic_params = self.critic_params)

    def update_episode(self) -> None:
        self.critic_losses.append(self.critic_loss_episode)
        self.actor_losses.append(self.actor_loss_episode)
        self.temperature_losses.append(self.temperature_loss_episode)
        self.td_errors.append(self.td_errors_episode)
        mean_temperature = jnp.mean(jnp.asarray(self.temperature_values_all_episode))
        self.temperature_values.append(mean_temperature)
        self.number_of_steps.append(self.number_of_steps_episode)

        # Log episode metrics to TensorBoard
        self.writer.add_scalar('Episode/CriticLoss', np.array(self.critic_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorLoss', np.array(self.actor_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/TemperatureLoss', np.array(self.temperature_loss_episode), self.episode_idx)
        self.writer.add_histogram('Episode/TDError', np.array(self.td_errors_episode), self.episode_idx)
        self.writer.add_scalar('Episode/MeanTemperature', np.array(mean_temperature), self.episode_idx)
        self.writer.add_scalar('Episode/Temperature', np.array(self.temperature), self.episode_idx)
        self.writer.add_scalar('Episode/NumberOfSteps', np.array(self.number_of_steps_episode), self.episode_idx)

        # Log weights
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

        
    def update(self) -> None:
        # Sample a batch of transitions from the buffer
        self.rng_key, subkey = jax.random.split(self.rng_key)
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(subkey)
        
        # Generate random noise for the batch
        mean_shape = (self.batch_size, self.action_dim)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        normal_distribution = jax.random.normal(subkey, mean_shape)
        states = jnp.asarray(states)
        normal_distribution = jnp.asarray(normal_distribution)
        
        # Rest of the update remains the same, just operating on batches now
        self.critic_params, self.critic_opt_state, critic_loss, td_errors, \
            self.actor_params, self.actor_opt_state, actor_loss, \
            self.temperature, self.temperature_opt_state, temperature_loss, \
            self.critic_target_params = self.update_sac_func(states=states,
                                                            actions=actions,
                                                            rewards=rewards,
                                                            next_states=next_states,
                                                            dones=dones,
                                                            weights=weights_buffer,
                                                            normal_dist=normal_distribution,
                                                            critic_params=self.critic_params,
                                                            critic_target_params=self.critic_target_params,
                                                            critic_opt_state=self.critic_opt_state,
                                                            actor_params=self.actor_params,
                                                            actor_opt_state=self.actor_opt_state,
                                                            temperature=self.temperature,
                                                            temperature_opt_state=self.temperature_opt_state)
        # Temporary
        self.temperature = jnp.clip(self.temperature, 0, 1)
        
        # Update priorities for the entire batch
        self.buffer.update_priorities(index, td_errors)
        
        self.critic_loss_episode += critic_loss
        self.actor_loss_episode += actor_loss
        self.temperature_loss_episode += temperature_loss
        self.td_errors_episode += td_errors
        self.temperature_values_all_episode.append(self.temperature)
        self.number_of_steps_episode += 1
            
        # Convert JAX arrays to NumPy arrays before logging
        critic_loss_np = np.array(critic_loss)
        actor_loss_np = np.array(actor_loss)
        temperature_loss_np = np.array(temperature_loss)
        td_errors_np = np.array(td_errors)
        temperature_np = np.array(self.temperature)

        # Log metrics to TensorBoard
        self.writer.add_scalar('Steps/CriticLoss', critic_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActorLoss', actor_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/TemperatureLoss', temperature_loss_np, self.step_idx)
        self.writer.add_histogram('Steps/TDError', td_errors_np, self.step_idx) # As batched
        self.writer.add_scalar('Steps/Temperature', temperature_np, self.step_idx)
        self.writer.add_scalar('Steps/NumberOfSteps', self.number_of_steps_episode, self.step_idx)

        # Log weights and biases for actor and critic
        for layer_name, layer_params in self.actor_params['params'].items():
            for param_name, param in layer_params.items():
                param_np = np.array(param)  # Convert to NumPy array
                self.writer.add_histogram(f'Actor/{layer_name}/{param_name}', param_np.flatten(), self.step_idx)

        for layer_name, layer_params in self.critic_params['params'].items():
            for param_name, param in layer_params.items():
                param_np = np.array(param)  # Convert to NumPy array
                self.writer.add_histogram(f'Critic/{layer_name}/{param_name}', param_np.flatten(), self.step_idx)

        # Update step index
        self.step_idx += 1
            
    def plotter(self):
        agent_plotter(self)

    def save(self, info: 'str') -> None:
        """
        Save the agent's state to disk.
        """
        # Extract last folder of self.save_path
        file_path = f'data/agent_saves/{self.model_name}/saves/soft-actor-critic_{info}.pkl'

        agent_state = {
            'seed': self.seed,
            'print_bool': self.print_bool,
            'dimensions': {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim_actor': self.hidden_dim_actor,
            'hidden_dim_critic': self.hidden_dim_critic,
            'std_min': self.std_min,
            'std_max': self.std_max,
            },
            'temperature': self.temperature,
            'target_entropy': self.target_entropy,
            'rng_key': jax.device_get(self.rng_key),
            'actor_params': jax.device_get(self.actor_params),
            'critic_params': jax.device_get(self.critic_params),
            'critic_target_params': jax.device_get(self.critic_target_params),
            'critic_opt_state': jax.device_get(self.critic_opt_state),
            'actor_opt_state': jax.device_get(self.actor_opt_state),
            'temperature_opt_state': jax.device_get(self.temperature_opt_state),
            'logging': {
                'critic_losses': self.critic_losses,
                'actor_losses': self.actor_losses,
                'temperature_losses': self.temperature_losses,
                'td_errors': self.td_errors,
                'temperature_values': self.temperature_values,
                'number_of_steps': self.number_of_steps,
                'critic_loss_episode': self.critic_loss_episode,
                'actor_loss_episode': self.actor_loss_episode,
                'temperature_loss_episode': self.temperature_loss_episode,
                'td_errors_episode': self.td_errors_episode,
                'temperature_values_all_episode': self.temperature_values_all_episode,
                'number_of_steps_episode': self.number_of_steps_episode,
            },
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'temperature_initial': self.temperature,
                'critic_grad_max_norm': self.critic_grad_max_norm,
                'critic_lr': self.critic_lr,
                'critic_weight_decay': self.critic_weight_decay,
                'actor_grad_max_norm': self.actor_grad_max_norm,
                'actor_lr': self.actor_lr,
                'actor_weight_decay': self.actor_weight_decay,
                'temperature_lr': self.temperature_lr,
                'temperature_grad_max_norm': self.temperature_grad_max_norm,
            },
            'model_name': self.model_name,
            'buffer_state': {
                'buffer': np.array(self.buffer.buffer),
                'priorities': np.array(self.buffer.priorities),
                'position': self.buffer.position,
                'n_step_buffer': np.array(self.buffer.n_step_buffer),
                'beta': self.buffer.beta,
                'static_config': self.buffer.static_config,
                'trajectory_length': self.trajectory_length,
            }
        }
        with open(file_path, 'wb') as f:
            pickle.dump(agent_state, f)
