from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pickle
import os

from src.agents.functions.networks import Actor
from src.agents.functions.networks import GaussianDoubleCritic as Critic
from src.agents.functions.buffers import PERInference as Buffer

from src.agents.functions.plotter import agent_plotter_sac_marl_ctde as agent_plotter
from src.agents.functions.soft_actor_critic_functions import (sample_actions, calculate_td_error,
                                                            actor_update, critic_update,
                                                            temperature_update, update_target_params,
                                                            gaussian_likelihood)
from src.agents.functions.marl_functions import (update_worker_actor, update_central_agent, update_marl)

class SAC_MARL_CTDE:
    '''
    Soft actor critic
    Multi-agent reinforcement learning (distributed training)
    Centralised training, decentralised execution
    i.e. shared critic and buffer, separate actors
    '''
    def __init__(self,
                 seed: int,
                 state_dim: int,
                 action_dim: int,
                 config: dict,
                 save_path: str = "results/VerticalRising-SAC-MARL-CTDE",
                 print_bool: bool = False):
        
        # Random variable
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(seed)

        # Save path
        self.save_path = save_path

        # Dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Number of agents
        self.number_of_workers = config['number_of_workers']
        
        # Initialise shared replay buffer
        self.buffer = Buffer(
            gamma=config['gamma'],
            alpha=config['buffer']['alpha'],
            beta=config['buffer']['beta'],
            beta_decay=config['buffer']['beta_decay'],
            buffer_size=config['buffer']['buffer_size'],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            trajectory_length=config['buffer']['trajectory_length'], # Nstep
            batch_size=config['batch_size']
        )

        # Store hyperparameters
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.hidden_dim_actor = config['hidden_dim_actor']
        self.hidden_dim_critic = config['hidden_dim_critic']
        self.std_min = config['std_min']
        self.std_max = config['std_max']
        self.tau = config['tau']

        # Target entropy
        self.target_entropy = -self.action_dim


        # Central agent
        self.central_actor_lr = config['central_agent']['central_actor_lr']
        self.central_critic_lr = config['central_agent']['central_critic_lr']
        self.central_temperature_lr = config['central_agent']['central_temperature_lr']
        self.actor_grad_max_norm_central = config['central_agent']['actor_grad_max_norm_central']
        self.critic_grad_max_norm_central = config['central_agent']['critic_grad_max_norm_central']
        self.temperature_grad_max_norm_central = config['central_agent']['temperature_grad_max_norm_central']

        # Worker agent
        self.worker_actor_lr = config['worker_agent']['worker_actor_lr']
        self.worker_temperature_lr = config['worker_agent']['worker_temperature_lr']
        self.actor_grad_max_norm_worker = config['worker_agent']['actor_grad_max_norm_worker']
        self.temperature_grad_max_norm_worker = config['worker_agent']['temperature_grad_max_norm_worker']
        
        
        # Initalise centralised actor and critic
        self.central_actor = Actor(action_dim=self.action_dim,
                                   hidden_dim=self.hidden_dim_actor)
        self.central_critic = Critic(state_dim=state_dim,
                                     action_dim=self.action_dim,
                                     hidden_dim=self.hidden_dim_critic)
        self.central_actor_params = self.central_actor.init(self.rng_key, jnp.zeros((1, self.state_dim)))
        self.central_critic_params = self.central_critic.init(self.rng_key, jnp.zeros((1, self.state_dim)), jnp.zeros((1, self.action_dim)))
        self.central_critic_target_params = self.central_critic_params

        # Initialise central learners
        self.central_actor_optimizer = optax.adam(learning_rate=self.central_actor_lr)
        self.central_critic_optimizer = optax.adam(learning_rate=self.central_critic_lr)
        self.central_actor_opt_state = self.central_actor_optimizer.init(self.central_actor_params)
        self.central_critic_opt_state = self.central_critic_optimizer.init(self.central_critic_params)
        self.central_temperature = config['central_agent']['central_temperature']
        self.central_temperature_optimizer = optax.adam(learning_rate=self.central_temperature_lr)
        self.central_temperature_opt_state = self.central_temperature_optimizer.init(self.central_temperature)

        # Initialise worker actors
        self.worker_actor = Actor(action_dim=self.action_dim,
                                   hidden_dim=self.hidden_dim_actor)
        self.worker_actor_optimizer = optax.adam(learning_rate=self.worker_actor_lr)
        self.worker_temperature = config['worker_agent']['worker_temperature']
        self.worker_temperature_optimizer = optax.adam(learning_rate=self.worker_temperature_lr)
        self.all_worker_actor_params = [self.worker_actor.init(self.rng_key, jnp.zeros((1, self.state_dim))) for _ in range(self.number_of_workers)]
        self.all_worker_actor_opt_state = [self.worker_actor_optimizer.init(self.all_worker_actor_params[i]) for i in range(self.number_of_workers)]
        self.all_worker_temperatures = [self.worker_temperature for _ in range(self.number_of_workers)]
        self.all_worker_temperature_opt_state = [self.worker_temperature_optimizer.init(self.all_worker_temperatures[i]) for i in range(self.number_of_workers)]

        # Create functions
        self.print_bool = print_bool
        self.create_funcs()

        # Logging
        self.central_critic_loss_episode = 0.0
        self.central_actor_loss_episode = 0.0
        self.central_temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.central_temperature_values_all_episode = []
        self.central_number_of_steps_episode = 0

        self.central_critic_losses = []
        self.central_actor_losses = []
        self.central_temperature_losses = []
        self.td_errors = []
        self.central_temperature_values = []
        self.central_number_of_steps = []

        self.actor_loss_workers_episode = []
        self.temperature_loss_workers_episode = []
        self.actor_loss_workers = []
        self.temperature_loss_workers = []
        self.worker_temperature_values = []

        # Track steps per episode for each worker
        self.number_of_steps_episode = [[] for _ in range(self.number_of_workers)]
        self.number_of_steps = [[] for _ in range(self.number_of_workers)]

    def create_funcs(self):
        if not self.print_bool:
            self.sample_actions_central_func = jax.jit(
                partial(sample_actions,
                        actor=self.central_actor,
                        std_min=self.std_min,
                        std_max=self.std_max,
                        print_bool=self.print_bool),
                static_argnames=['actor', 'std_min', 'std_max', 'print_bool']
            )

            self.sample_actions_worker_func = jax.jit(
                partial(sample_actions,
                        actor=self.worker_actor,
                        std_min=self.std_min,
                        std_max=self.std_max,
                        print_bool=self.print_bool),
                static_argnames=['actor', 'std_min', 'std_max', 'print_bool']
            )

            self.actor_update_central_func = jax.jit(
                partial(actor_update,
                        actor_optimizer = self.central_actor_optimizer,
                        sample_actions_func = self.sample_actions_central_func,
                        critic = self.central_critic,
                        actor_grad_max_norm = self.actor_grad_max_norm_central,
                        actor = self.central_actor,
                        std_min = self.std_min,
                        std_max = self.std_max,
                        print_bool = self.print_bool),
                static_argnames=['sample_actions_func', 'critic', 'actor', 'std_min', 'std_max', 'print_bool']
            )

            self.actor_update_worker_func = jax.jit(
                partial(actor_update,
                        actor_optimizer = self.worker_actor_optimizer,
                        sample_actions_func = self.sample_actions_worker_func,
                        critic = self.central_critic,
                        actor_grad_max_norm = self.actor_grad_max_norm_worker,
                        actor = self.worker_actor,
                        std_min = self.std_min,
                        std_max = self.std_max,
                        print_bool = self.print_bool),
                static_argnames=['sample_actions_func', 'critic', 'actor', 'std_min', 'std_max', 'print_bool']
            )

            self.temperature_update_central_func = jax.jit(
                partial(temperature_update,
                        temperature_optimizer = self.central_temperature_optimizer,
                        target_entropy = self.target_entropy,
                        temperature_grad_max_norm = self.temperature_grad_max_norm_central,
                        print_bool = self.print_bool),
                static_argnames=['temperature_optimizer', 'target_entropy', 'temperature_grad_max_norm', 'print_bool']
            )

            self.temperature_update_worker_func = jax.jit(
                partial(temperature_update,
                        temperature_optimizer = self.worker_temperature_optimizer,
                        target_entropy = self.target_entropy,
                        temperature_grad_max_norm = self.temperature_grad_max_norm_worker,
                        print_bool = self.print_bool),
                static_argnames=['temperature_optimizer', 'target_entropy', 'temperature_grad_max_norm', 'print_bool']
            )

            self.update_worker_actor_func = jax.jit(
                partial(update_worker_actor,
                        sample_actions_func = self.sample_actions_worker_func,
                        actor_update_func = self.actor_update_worker_func,
                        temperature_update_func = self.temperature_update_worker_func),
                static_argnames=['sample_actions_func', 'actor_update_func', 'temperature_update_func']
            )

            self.calculate_td_error_func = jax.jit(
                partial(calculate_td_error,
                        critic=self.central_critic,
                        gamma=self.gamma,
                        print_bool=self.print_bool),
                static_argnames=['critic', 'gamma', 'print_bool']
            )

            self.critic_update_central_func = jax.jit(
                partial(critic_update,
                        critic_optimizer = self.central_critic_optimizer,
                        calculate_td_error_fn = self.calculate_td_error_func,
                        critic_grad_max_norm = self.critic_grad_max_norm_central,
                        print_bool = self.print_bool),
                static_argnames=['critic_optimizer', 'calculate_td_error_fn', 'critic_grad_max_norm', 'print_bool']
            )

            self.update_target_params_func = jax.jit(
                partial(update_target_params,
                        tau=self.tau,
                        print_bool=self.print_bool),
                static_argnames=['tau', 'print_bool']
            )

            self.update_central_agent_func = jax.jit(
                partial(update_central_agent,
                        sample_actions_func = self.sample_actions_central_func,
                        critic_update_func = self.critic_update_central_func,
                        actor_update_func = self.actor_update_central_func,
                        temperature_update_func = self.temperature_update_central_func,
                        update_target_params_critic_func = self.update_target_params_func,
                        print_bool = self.print_bool),
                static_argnames=['sample_actions_func', 'critic_update_func', 'actor_update_func', 'temperature_update_func', 'update_target_params_critic_func', 'print_bool']
            )

            self.update_marl_func = jax.jit(
                partial(update_marl,
                        update_worker_actor_func = self.update_worker_actor_func,
                        update_central_agent_func = self.update_central_agent_func),
                static_argnames=['update_worker_actor_func', 'update_central_agent_func']
            )

            

        else:
            self.sample_actions_central_func = lambda states, actor_params, normal_dist: sample_actions(states, actor_params, normal_dist, self.central_actor, self.std_min, self.std_max, self.print_bool)
            self.sample_actions_worker_func = lambda states, actor_params, normal_dist: sample_actions(states, actor_params, normal_dist, self.worker_actor, self.std_min, self.std_max, self.print_bool)
            self.actor_update_central_func = lambda  states, temperature_central, central_critic_params, \
                central_actor_params, central_actor_opt_state, normal_distribution: \
                actor_update(self.central_actor_optimizer,
                             self.sample_actions_central_func,
                             states, temperature_central,
                             self.central_critic, central_critic_params,
                             central_actor_params, central_actor_opt_state,
                             self.actor_grad_max_norm_central, normal_distribution,
                             self.central_actor, self.std_min, self.std_max, self.print_bool)
            
            self.actor_update_worker_func = lambda states, temperature_worker, central_critic_params, \
                worker_actor_params, worker_actor_opt_state, normal_distribution: \
                actor_update(self.worker_actor_optimizer,
                             self.sample_actions_worker_func,
                             states, temperature_worker, self.central_critic, central_critic_params,
                             worker_actor_params, worker_actor_opt_state,
                             self.actor_grad_max_norm_worker, normal_distribution,
                             self.worker_actor, self.std_min, self.std_max, self.print_bool)
            
            self.temperature_update_central_func = lambda temperature_central, temperature_opt_state_central, \
                log_probs : temperature_update(self.central_temperature_optimizer,
                                               temperature_central,
                                               temperature_opt_state_central,
                                               log_probs,
                                               self.target_entropy,
                                               self.temperature_grad_max_norm_central,
                                               self.print_bool)
            
            self.temperature_update_worker_func = lambda temperature_worker, temperature_opt_state_worker, \
                log_probs : temperature_update(self.worker_temperature_optimizer,
                                               temperature_worker,
                                               temperature_opt_state_worker,
                                               log_probs,
                                               self.target_entropy,
                                               self.temperature_grad_max_norm_worker,
                                               self.print_bool)
            
            self.update_worker_actor_func = lambda states, actor_params_worker, actor_opt_state_worker, \
                temperature_worker, temperature_opt_state_worker, normal_distribution, \
                central_critic_params : update_worker_actor(states, actor_params_worker, actor_opt_state_worker,
                                                           temperature_worker, temperature_opt_state_worker,
                                                           normal_distribution, central_critic_params,
                                                           self.sample_actions_worker_func,
                                                           self.actor_update_worker_func,
                                                           self.temperature_update_worker_func)
            self.calculate_td_error_func = lambda states, actions, rewards, next_states, dones, temperature, \
                central_critic_params, central_critic_target_params, next_log_policy: \
                    calculate_td_error(states, actions, rewards, next_states, dones, temperature, \
                                       self.gamma, central_critic_params, central_critic_target_params, \
                                        self.central_critic, next_log_policy, self.print_bool)
                        

            self.critic_update_central_func = lambda states, actions, rewards, next_states, dones, weights, critic_params, \
                critic_opt_state, critic_target_params, temperature, next_log_policy: \
                    critic_update(self.central_critic_optimizer, self.calculate_td_error_func, states, actions, rewards, next_states, dones, weights,\
                                   critic_params, critic_opt_state,  self.critic_grad_max_norm_central, critic_target_params, temperature, next_log_policy, self.print_bool)                                                                                                             

            self.update_target_params_func = lambda target_params, params: update_target_params(self.tau, target_params, params, self.print_bool)
            
            self.update_central_agent_func = lambda states, actions, rewards, next_states, done, weights, \
                normal_distribution, central_critic_params, central_critic_target_params, central_critic_opt_state, \
                central_actor_params, central_actor_opt_state, central_temperature, central_temperature_opt_state: \
                update_central_agent(self.sample_actions_central_func,
                                     self.critic_update_central_func,
                                     self.actor_update_central_func,
                                     self.temperature_update_central_func,
                                     self.update_target_params_func,
                                     states, actions, rewards, next_states, done, weights, normal_distribution,
                                     central_critic_params,
                                     central_critic_target_params,
                                     central_critic_opt_state,
                                     central_actor_params,
                                     central_actor_opt_state,
                                     central_temperature,
                                     central_temperature_opt_state,
                                     self.print_bool)
            
            self.update_marl_func = lambda all_workers_actor_params, all_workers_actor_opt_state, all_workers_temperatures, \
                all_workers_temperature_opt_state, states, actions, rewards, next_states, dones, weights, normal_distributions, \
                central_critic_params, central_critic_target_params, central_critic_opt_state, \
                central_actor_params, central_actor_opt_state, central_temperature, central_temperature_opt_state: \
                update_marl(self.update_worker_actor_func, self.update_central_agent_func,
                            all_workers_actor_params, all_workers_actor_opt_state, all_workers_temperatures, all_workers_temperature_opt_state,
                            states, actions, rewards, next_states, dones, weights, normal_distributions,
                            central_critic_params, central_critic_target_params, central_critic_opt_state,
                            central_actor_params, central_actor_opt_state, central_temperature, central_temperature_opt_state)
            
            
    def reset(self, rng_key: jnp.ndarray) -> None:
        self.rng_key = rng_key
        self.central_critic_losses = []
        self.central_actor_losses = []
        self.central_temperature_losses = []
        self.td_errors = []
        self.central_temperature_values = []
        self.central_number_of_steps = []
        self.buffer.reset(rng_key)
        self.create_funcs()
        
        # Central agent episode variables
        self.central_critic_loss_episode = 0.0
        self.central_actor_loss_episode = 0.0
        self.central_temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.central_temperature_values_all_episode = []
        self.central_number_of_steps_episode = 0

        # Worker episode variables
        self.actor_loss_workers_episode = []
        self.temperature_loss_workers_episode = []
        self.actor_loss_workers = []
        self.temperature_loss_workers = []
        self.worker_temperature_values = []
        
        # Track steps per episode for each worker
        self.number_of_steps_episode = [[] for _ in range(self.number_of_workers)]
        self.number_of_steps = [[] for _ in range(self.number_of_workers)]

    def update_episode(self) -> None:
        """
        Update episode-level statistics and reset episode-specific variables
        """
        # Calculate episode averages for central agent
        if self.central_number_of_steps_episode > 0:  # Avoid division by zero
            self.central_critic_losses.append(self.central_critic_loss_episode / self.central_number_of_steps_episode)
            self.central_actor_losses.append(self.central_actor_loss_episode / self.central_number_of_steps_episode)
            self.central_temperature_losses.append(self.central_temperature_loss_episode / self.central_number_of_steps_episode)
            self.td_errors.append(self.td_errors_episode / self.central_number_of_steps_episode)
            self.central_temperature_values.append(np.mean(self.central_temperature_values_all_episode))
            self.central_number_of_steps.append(self.central_number_of_steps_episode)

        # Update worker statistics - calculate episode averages
        if self.actor_loss_workers_episode:  # Check if we have any steps in this episode
            actor_loss_workers_array = jnp.array(self.actor_loss_workers_episode)
            temperature_loss_workers_array = jnp.array(self.temperature_loss_workers_episode)
            
            # Calculate mean losses for this episode
            actor_loss_workers = jnp.mean(actor_loss_workers_array, axis=0)  # Average over steps
            temperature_loss_workers = jnp.mean(temperature_loss_workers_array, axis=0)  # Average over steps

            self.actor_loss_workers.append(actor_loss_workers)
            self.temperature_loss_workers.append(temperature_loss_workers)

        # Store steps for each worker
        for i in range(self.number_of_workers):
            self.number_of_steps[i].append(len(self.number_of_steps_episode[i]))
            self.number_of_steps_episode[i] = []  # Reset for next episode

        # Reset episode-specific variables for central agent
        self.central_critic_loss_episode = 0
        self.central_actor_loss_episode = 0
        self.central_temperature_loss_episode = 0
        self.td_errors_episode = 0
        self.central_temperature_values_all_episode = []
        self.central_number_of_steps_episode = 0

        # Reset episode-specific variables for workers
        self.actor_loss_workers_episode = []
        self.temperature_loss_workers_episode = []
        self.worker_temperature_values.append(self.all_worker_temperatures)

    def plotter(self):
        agent_plotter(self)

    def log_probability_next_action(self, state: jnp.ndarray) -> jnp.ndarray:  # Centralised log probability of next action
        batch_size = state.shape[0]
        mean_shape = (batch_size, self.action_dim)
        normal_distribution = jax.random.normal(self.rng_key, mean_shape)
        states = jnp.asarray(state)
        normal_distribution = jnp.asarray(normal_distribution)
        actions, mean, std = self.sample_actions_central_func(states=states,
                                                    actor_params=self.central_actor_params,
                                                    normal_dist=normal_distribution)
        log_probability = gaussian_likelihood(actions, mean, std)
        return log_probability

    def calculate_td_error(self,
                            states: jnp.ndarray,
                            actions: jnp.ndarray,
                            rewards: jnp.ndarray,
                            next_states: jnp.ndarray,
                            dones: jnp.ndarray) -> jnp.ndarray:  # Centralised TD error
        next_log_policy = self.log_probability_next_action(next_states)
        td_errors = self.calculate_td_error_func(states = states,
                                                 actions = actions,
                                                 rewards = rewards,
                                                 next_states = next_states,
                                                 dones = dones,
                                                 temperature = self.central_temperature,
                                                 critic_params = self.central_critic_params,
                                                 critic_target_params = self.central_critic_target_params,
                                                 next_log_policy = next_log_policy)
        return td_errors
    
    def select_actions_from_all_workers(self, states):
        """
        Select actions from all workers and the central agent.
        
        Args:
            states: List of states, one for each worker and the central agent
            
        Returns:
            List of actions, one for each worker and the central agent
        """
        # Initialize actions list with None values
        actions = [None] * (self.number_of_workers + 1)
        
        # Get a batch of random keys for each worker
        keys = jax.random.split(self.rng_key, self.number_of_workers + 1)
        self.rng_key = keys[0]  # Update main key
        
        # Select actions for each worker
        for i in range(self.number_of_workers):
            subkey = keys[i+1]  # Use i+1 since we used keys[0] for main key
            normal_distribution = jax.random.normal(subkey, (1, self.action_dim))  # Changed batch_size to 1
            state = jnp.expand_dims(states[i], 0)
            action, _, _ = self.sample_actions_worker_func(
                states=jnp.asarray(state),
                actor_params=self.all_worker_actor_params[i],
                normal_dist=jnp.asarray(normal_distribution)
            )
            actions[i] = jnp.squeeze(action)  # Remove batch dimension
        
        # Select action for central agent
        normal_distribution = jax.random.normal(self.rng_key, (1, self.action_dim))  # Changed batch_size to 1
        state = jnp.expand_dims(states[-1], 0)
        central_action, _, _ = self.sample_actions_central_func(
            states=jnp.asarray(state),
            actor_params=self.central_actor_params,
            normal_dist=jnp.asarray(normal_distribution)
        )
        actions[-1] = jnp.squeeze(central_action)  # Remove batch dimension
        
        return actions

    def select_actions_no_stochatic(self, state: jnp.ndarray) -> jnp.ndarray:  # From central actor
        mean, std_0_1 = self.central_actor.apply(self.central_actor_params, state)
        return mean

    def select_actions(self, state: jnp.ndarray) -> jnp.ndarray:  # From central actor
        self.rng_key, subkey = jax.random.split(self.rng_key)
        normal_distribution = jax.random.normal(subkey, (self.batch_size, self.action_dim))
        actions, _, _ = self.sample_actions_central_func(
            states=jnp.asarray(state),
            actor_params=self.central_actor_params,
            normal_dist=jnp.asarray(normal_distribution)
        )
        return actions

    def update(self) -> None:
        number_of_agents = self.number_of_workers + 1

        # Initialize arrays to store batch data
        states_all_agents = []
        actions_all_agents = []
        rewards_all_agents = []
        next_states_all_agents = []
        dones_all_agents = []
        weights_all_agents = []
        indices_all_agents = []
        normal_distributions_all_agents = []

        # Sample from buffer for each agent
        for i in range(number_of_agents):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            # Sample from buffer
            states, actions, rewards, next_states, dones, indices, weights = self.buffer(self.rng_key)
            
            # Append to lists
            states_all_agents.append(states)
            actions_all_agents.append(actions)
            rewards_all_agents.append(rewards)
            next_states_all_agents.append(next_states)
            dones_all_agents.append(dones)
            indices_all_agents.append(indices)
            weights_all_agents.append(weights)
            
            # Generate normal distributions for each agent
            normal_distributions_all_agents.append(
                jax.random.normal(subkey, (self.batch_size, self.action_dim))
            )

        # Convert lists to JAX arrays
        states_all_agents = jnp.array(states_all_agents)
        actions_all_agents = jnp.array(actions_all_agents)
        rewards_all_agents = jnp.array(rewards_all_agents)
        next_states_all_agents = jnp.array(next_states_all_agents)
        dones_all_agents = jnp.array(dones_all_agents)
        indices_all_agents = jnp.array(indices_all_agents)
        weights_all_agents = jnp.array(weights_all_agents)
        normal_distributions_all_agents = jnp.array(normal_distributions_all_agents)

        # Update all agents
        self.central_critic_params, self.central_critic_opt_state, central_critic_loss, td_errors, \
            self.central_actor_params, self.central_actor_opt_state, central_actor_loss, \
                self.central_temperature, self.central_temperature_opt_state, central_temperature_loss, \
                    self.central_critic_target_params, self.all_worker_actor_params, self.all_worker_actor_opt_state, \
                        self.all_worker_temperatures, self.all_worker_temperature_opt_state, \
                              actor_loss_workers, temperature_loss_workers = \
                                self.update_marl_func(
                                    all_workers_actor_params=self.all_worker_actor_params,
                                    all_workers_actor_opt_state=self.all_worker_actor_opt_state,
                                    all_workers_temperatures=self.all_worker_temperatures,
                                    all_workers_temperature_opt_state=self.all_worker_temperature_opt_state,
                                    states=states_all_agents,
                                    actions=actions_all_agents,
                                    rewards=rewards_all_agents,
                                    next_states=next_states_all_agents,
                                    dones=dones_all_agents,
                                    weights=weights_all_agents,
                                    normal_distributions=normal_distributions_all_agents,
                                    central_critic_params=self.central_critic_params,
                                    central_critic_target_params=self.central_critic_target_params,
                                    central_critic_opt_state=self.central_critic_opt_state,
                                    central_actor_params=self.central_actor_params,
                                    central_actor_opt_state=self.central_actor_opt_state,
                                    central_temperature=self.central_temperature,
                                    central_temperature_opt_state=self.central_temperature_opt_state)
        
        # Update buffer priorities
        self.buffer.update_priorities(indices_all_agents, td_errors)

        # Update logging information
        self.central_critic_loss_episode += central_critic_loss
        self.central_actor_loss_episode += central_actor_loss
        self.central_temperature_loss_episode += central_temperature_loss
        self.td_errors_episode += td_errors
        self.central_temperature_values_all_episode.append(self.central_temperature)
        self.central_number_of_steps_episode += 1

        self.actor_loss_workers_episode.append(actor_loss_workers)
        self.temperature_loss_workers_episode.append(temperature_loss_workers)

        # Track step for each worker
        for i in range(self.number_of_workers):
            self.number_of_steps_episode[i].append(1)  # Increment step count for each worker

    def save(self, info: 'str') -> None:
        file_path_final_folder = self.save_path.split("/")[-2]
        file_path = os.path.join('..', 'data', 'agents_saves', file_path_final_folder, f'soft-actor-critic-marl-ctde_{info}.pkl')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        agent_state = {
                'seed': self.seed,
                'rng_key': jax.device_get(self.rng_key),
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'buffer_params': {
                    'gamma': self.buffer.static_config['gamma'],
                    'alpha': self.buffer.static_config['alpha'],
                    'beta': self.buffer.beta,
                    'beta_decay': self.buffer.static_config['beta_decay'],
                    'buffer_size': self.buffer.static_config['buffer_size'],
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'batch_size': self.buffer.static_config['batch_size'],
                    'buffer': np.array(self.buffer.buffer),
                    'priorities': np.array(self.buffer.priorities),
                    'position': self.buffer.position,
                    'n_step_buffer': np.array(self.buffer.n_step_buffer),
                    'beta': self.buffer.beta,
                    'static_config': self.buffer.static_config                    
                },
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'hidden_dim_actor': self.hidden_dim_actor,
                'hidden_dim_critic': self.hidden_dim_critic,
                'std_min': self.std_min,
                'std_max': self.std_max,
                'tau': self.tau,
                'target_entropy': self.target_entropy,
                'central_agent' : {
                    'central_actor_lr': self.central_actor_lr,
                    'central_critic_lr': self.central_critic_lr,
                    'central_temperature_lr': self.central_temperature_lr,
                    'critic_grad_max_norm_central': self.critic_grad_max_norm_central,
                    'temperature_grad_max_norm_central': self.temperature_grad_max_norm_central,
                    'actor_grad_max_norm_central': self.actor_grad_max_norm_central,
                    'central_actor_params': jax.device_get(self.central_actor_params),
                    'central_critic_params': jax.device_get(self.central_critic_params),
                    'central_critic_target_params': jax.device_get(self.central_critic_target_params),
                    'central_actor_opt_state': jax.device_get(self.central_actor_opt_state),
                    'central_critic_opt_state': jax.device_get(self.central_critic_opt_state),
                    'central_temperature_opt_state': jax.device_get(self.central_temperature_opt_state),
                    'central_temperature': self.central_temperature
                },
                'number_of_workers': self.number_of_workers,
                'worker_agent' : {
                    'worker_actor_lr': self.worker_actor_lr,
                    'worker_temperature_lr': self.worker_temperature_lr,
                    'actor_grad_max_norm_worker': self.actor_grad_max_norm_worker,
                    'temperature_grad_max_norm_worker': self.temperature_grad_max_norm_worker,
                    'all_worker_actor_params': jax.device_get(self.all_worker_actor_params),
                    'all_worker_actor_opt_state': jax.device_get(self.all_worker_actor_opt_state),
                    'all_worker_temperatures': self.all_worker_temperatures,
                    'all_worker_temperature_opt_state': jax.device_get(self.all_worker_temperature_opt_state),
                },
                'print_bool': self.print_bool,
                'save_path': self.save_path,
                'logging' : {
                    'central_critic_loss_episode': self.central_critic_loss_episode,
                    'central_actor_loss_episode': self.central_actor_loss_episode,
                    'central_temperature_loss_episode': self.central_temperature_loss_episode,
                    'td_errors_episode': self.td_errors_episode,
                    'central_temperature_values_all_episode': self.central_temperature_values_all_episode,
                    'central_number_of_steps_episode': self.central_number_of_steps_episode,
                    'central_critic_losses': self.central_critic_losses,
                    'central_actor_losses': self.central_actor_losses,
                    'central_temperature_losses': self.central_temperature_losses,
                    'td_errors': self.td_errors,
                    'central_temperature_values': self.central_temperature_values,
                    'central_number_of_steps': self.central_number_of_steps,
                    'actor_loss_workers': self.actor_loss_workers,
                    'temperature_loss_workers': self.temperature_loss_workers,
                    'worker_temperature_values': self.worker_temperature_values,
                    'actor_loss_workers_episode': self.actor_loss_workers_episode,
                    'temperature_loss_workers_episode': self.temperature_loss_workers_episode
                },                
            }

        with open(file_path, 'wb') as f:
            pickle.dump(agent_state, f)