import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.td3_functions import lambda_compile_td3
from src.agents.functions.plotter import agent_plotter_td3
from src.agents.functions.buffers import PERBuffer
from src.agents.functions.networks import DoubleCritic
from src.agents.functions.networks import ClassicalActor as Actor

class TD3:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 flight_phase: str,
                 # Dimensions
                 hidden_dim_actor: int,
                 number_of_hidden_layers_actor: int,
                 hidden_dim_critic: int,
                 number_of_hidden_layers_critic: int,
                 # Hyper-parameters
                 gamma: float,
                 tau: float,
                 alpha_buffer: float,
                 beta_buffer: float,
                 beta_decay_buffer: float,
                 buffer_size: int,
                 trajectory_length: int,
                 batch_size: int,
                 # Learning rates
                 critic_learning_rate: float,
                 actor_learning_rate: float,
                 # Grad max norms
                 critic_grad_max_norm: float,
                 actor_grad_max_norm: float,
                 # TD3 specific
                 policy_noise: float, # STD of Gaussian noise added to the actions
                 noise_clip: float, # Clipping value for the noise i.e. max value
                 policy_delay: int): # Number of steps actor is update relative to critic
        
        self.rng_key = jax.random.PRNGKey(0)
        
        self.save_path = f'results/TD3/{flight_phase}/'
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
        self.writer = SummaryWriter(log_dir=f'data/agent_saves/TD3/{flight_phase}/runs/{self.run_id}')

        # Initialize networks
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

        # Store hyperparameters
        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm
        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_max_norm = actor_grad_max_norm
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        self.hidden_dim_actor = hidden_dim_actor
        self.number_of_hidden_layers_actor = number_of_hidden_layers_actor
        self.hidden_dim_critic = hidden_dim_critic
        self.number_of_hidden_layers_critic = number_of_hidden_layers_critic

        self.alpha_buffer = alpha_buffer
        self.beta_buffer = beta_buffer
        self.beta_decay_buffer = beta_decay_buffer
        self.buffer_size = buffer_size
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Compile update functions
        self.critic_optimiser = optax.adam(learning_rate=self.critic_learning_rate)
        self.actor_optimiser = optax.adam(learning_rate=self.actor_learning_rate)
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda = lambda_compile_td3(
            critic_optimiser=self.critic_optimiser,
            critic=self.critic,
            critic_grad_max_norm=self.critic_grad_max_norm,
            actor_optimiser=self.actor_optimiser,
            actor=self.actor,
            actor_grad_max_norm=self.actor_grad_max_norm,
            gamma=self.gamma,
            tau=self.tau,
            policy_delay=self.policy_delay
        )

        # Logging
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.td_errors = []
        self.number_of_steps = []

        self.use_prioritized_sampling()

        self.name = 'TD3'
        self.critic_warm_up_step_idx = 0

    def re_init_actor(self, new_actor, new_actor_params):
        self.actor = new_actor
        self.actor_params = new_actor_params
        self.actor_opt_state = optax.adam(learning_rate=self.actor_learning_rate).init(self.actor_params)
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda = lambda_compile_td3(
            critic_optimiser=self.critic_optimiser,
            critic=self.critic,
            critic_grad_max_norm=self.critic_grad_max_norm,
            actor_optimiser=self.actor_optimiser,
            actor=self.actor,
            actor_grad_max_norm=self.actor_grad_max_norm,
            gamma=self.gamma,
            tau=self.tau,
            policy_delay=self.policy_delay
        )

    def reset(self):
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.td_errors = []
        self.number_of_steps = []
        self.rng_key = jax.random.PRNGKey(0)
        self.buffer.reset()
        self.critic_warm_up_step_idx = 0

    def get_subkey(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey
    
    def get_normal_distributions_batched(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.batch_size, self.action_dim)))
        return normal_distribution
    
    def get_normal_distribution(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.action_dim, )))
        return normal_distribution

    def select_actions(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select actions with exploration noise."""
        actions = self.actor.apply(self.actor_params, state)
        noise = jnp.clip(
            self.get_normal_distribution()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )
        return jnp.squeeze(jnp.clip(actions + noise, -1, 1))

    def select_actions_no_stochastic(self, state: jnp.ndarray) -> jnp.ndarray:
        """Select actions without exploration noise."""
        action = self.actor.apply(self.actor_params, state)
        return jnp.clip(action, -1, 1)

    def calculate_td_error(self,
                          state: jnp.ndarray,
                          action: jnp.ndarray,
                          reward: jnp.ndarray,
                          next_state: jnp.ndarray,
                          done: jnp.ndarray) -> jnp.ndarray:
        next_action = self.actor.apply(self.actor_params, next_state)
        noise = jnp.clip(
            self.get_normal_distribution()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )
        next_action = next_action + noise
        return self.calculate_td_error_lambda(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            dones=done,
            critic_params=self.critic_params,
            critic_target_params=self.critic_target_params,
            next_actions=next_action
        )
    
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
        next_action_means = jax.vmap(self.actor.apply, in_axes=(None, 0))(self.actor_params, next_states)
        
        # get normal distributions in batch with proper key splitting
        self.rng_key, *subkeys = jax.random.split(self.rng_key, len(next_states) + 1)
        normal_distributions = jax.vmap(lambda key: jax.random.normal(key, (self.action_dim,)))(jnp.array(subkeys))
        
        # use vmap for next actions
        next_actions = jax.vmap(lambda mean, normal: normal * self.policy_noise + mean)(
            next_action_means, normal_distributions
        )

        def td_error_calc(state, action, reward, next_state, done, next_action):
            return self.calculate_td_error_lambda(
                states=state,
                actions=action,
                rewards=reward,
                next_states=next_state,
                dones=done,
                critic_params=self.critic_params,
                critic_target_params=self.critic_target_params,
                next_actions=next_action
            )
        # Calculate TD errors for all transitions using vmap
        td_errors = jax.vmap(td_error_calc)(states, actions, rewards, next_states, dones, next_actions)
        
        return td_errors
    
    def critic_warm_up_step(self):
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(self.get_subkey())

        clipped_noise = jnp.clip(
            self.get_normal_distributions_batched()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )

        self.critic_params, self.critic_opt_state, self.critic_target_params, critic_loss_warm_up = self.critic_warm_up_update_lambda(
            actor_params = self.actor_params,
            states = states,
            actions = actions,
            rewards = rewards,
            next_states = next_states,
            dones = dones,
            buffer_weights = weights_buffer,
            critic_params = self.critic_params,
            critic_target_params = self.critic_target_params,
            critic_opt_state = self.critic_opt_state,
            clipped_noise = clipped_noise)
        
        self.writer.add_scalar('CriticWarmUp/Loss', np.array(critic_loss_warm_up), self.critic_warm_up_step_idx)
        self.critic_warm_up_step_idx += 1
        return critic_loss_warm_up

    def update_episode(self):
        self.critic_losses.append(self.critic_loss_episode)
        self.actor_losses.append(self.actor_loss_episode)
        self.td_errors.append(self.td_errors_episode)
        self.number_of_steps.append(self.number_of_steps_episode)

        # Log episode metrics
        self.writer.add_scalar('Episode/CriticLoss', np.array(self.critic_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorLoss', np.array(self.actor_loss_episode), self.episode_idx)
        self.writer.add_histogram('Episode/TDError', np.array(self.td_errors_episode), self.episode_idx)
        self.writer.add_scalar('Episode/NumberOfSteps', np.array(self.number_of_steps_episode), self.episode_idx)

        # Reset episode metrics
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.episode_idx += 1

    def update(self):
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(self.get_subkey())

        clipped_noise = jnp.clip(
            self.get_normal_distributions_batched()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )

        self.critic_params, self.critic_opt_state, critic_loss, td_errors, \
        self.actor_params, self.actor_opt_state, actor_loss, \
        self.critic_target_params = self.update_function(
            actor_params=self.actor_params,
            actor_opt_state=self.actor_opt_state,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            buffer_weights=weights_buffer,
            critic_params=self.critic_params,
            critic_target_params=self.critic_target_params,
            critic_opt_state=self.critic_opt_state,
            clipped_noise = clipped_noise,
            step=self.step_idx
        )

        # Update buffer priorities
        self.buffer.update_priorities(index, td_errors)

        # Update metrics
        self.critic_loss_episode += critic_loss
        self.actor_loss_episode += actor_loss
        self.td_errors_episode += td_errors
        self.number_of_steps_episode += 1
        self.step_idx += 1

        # Log step metrics
        self.writer.add_scalar('Steps/CriticLoss', np.array(critic_loss), self.step_idx)
        self.writer.add_scalar('Steps/ActorLoss', np.array(actor_loss), self.step_idx)
        self.writer.add_scalar('Steps/TDError/mean', np.mean(np.array(td_errors)), self.step_idx)
        self.writer.add_scalar('Steps/TDError/std', np.std(np.array(td_errors)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/mean', np.mean(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/std', np.std(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/max', np.max(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/min', np.min(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/mean', np.mean(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/std', np.std(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/max', np.max(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/min', np.min(np.array(clipped_noise)), self.step_idx)

    def plotter(self):
        agent_plotter_td3(self)

    def save(self):
        file_path = f'data/agent_saves/TD3/{self.flight_phase}/saves/td3.pkl'
        agent_state = {
            'inputs': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'flight_phase': self.flight_phase,
                'hidden_dim_actor': self.hidden_dim_actor,
                'number_of_hidden_layers_actor': self.number_of_hidden_layers_actor,
                'hidden_dim_critic': self.hidden_dim_critic,
                'number_of_hidden_layers_critic': self.number_of_hidden_layers_critic,
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha_buffer': self.alpha_buffer,
                'beta_buffer': self.beta_buffer,
                'beta_decay_buffer': self.beta_decay_buffer,
                'buffer_size': self.buffer_size,
                'trajectory_length': self.trajectory_length,
                'batch_size': self.batch_size,
                'critic_learning_rate': self.critic_learning_rate,
                'actor_learning_rate': self.actor_learning_rate,
                'critic_grad_max_norm': self.critic_grad_max_norm,
                'actor_grad_max_norm': self.actor_grad_max_norm,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_delay': self.policy_delay
            },
            'misc': {
                'rng_key': self.rng_key,
                'run_id': self.run_id
            },
            'logging': {
                'critic_loss_episode': self.critic_loss_episode,
                'actor_loss_episode': self.actor_loss_episode,
                'td_errors_episode': self.td_errors_episode,
                'number_of_steps_episode': self.number_of_steps_episode,
                'episode_idx': self.episode_idx,
                'step_idx': self.step_idx,
                'critic_losses': self.critic_losses,
                'actor_losses': self.actor_losses,
                'td_errors': self.td_errors,
                'number_of_steps': self.number_of_steps
            },
            'update': {
                'critic_params': self.critic_params,
                'critic_opt_state': self.critic_opt_state,
                'critic_target_params': self.critic_target_params,
                'actor_params': self.actor_params,
                'actor_opt_state': self.actor_opt_state,
                'buffer': self.buffer
            }
        }

        with open(file_path, 'wb') as f:
            pickle.dump(agent_state, f)

    # PER Buffer control methods
    def use_prioritized_sampling(self):
        """Switch the buffer to use prioritized experience replay"""
        self.buffer.set_uniform_sampling(False)
        
    def use_uniform_sampling(self):
        """Switch the buffer to use uniform sampling"""
        self.buffer.set_uniform_sampling(True)
        
    def toggle_sampling_mode(self):
        """Toggle between prioritized and uniform sampling"""
        current = self.buffer.is_using_uniform_sampling()
        self.buffer.set_uniform_sampling(not current)
        return not current
        
    def get_sampling_mode(self):
        """Get current sampling mode (True for uniform, False for prioritized)"""
        return self.buffer.is_using_uniform_sampling() 