import os
import jax
import time
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.trainers.trainers import TrainerSAC
from configs.agent_config import config_subsonic, config_supersonic, config_flip_over_boostbackburn, config_ballistic_arc_descent
from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.functions.load_agent import load_sac
from src.envs.rl.parallel_env_wrapped_rl import ParallelRocketEnv
from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.critic_pre_train.pre_train_critic import pre_train_critic_from_pso_experiences
from src.envs.supervisory.agent_load_supervisory import load_supervisory_actor
from src.agents.functions.soft_actor_critic_functions import gaussian_likelihood

class TrainerSAC_Parallel(TrainerSAC):
    def __init__(self, 
                 env,
                 agent,
                 flight_phase: str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 load_buffer_from_experiences_bool: bool = False,
                 update_agent_every_n_steps: int = 10,
                 num_parallel_envs: int = 4):
        super().__init__(env, agent, flight_phase, num_episodes, save_interval, 
                         critic_warm_up_steps, load_buffer_from_experiences_bool,
                         update_agent_every_n_steps)
        self.num_parallel_envs = num_parallel_envs
        
        # Create parallel environment
        self.parallel_env = ParallelRocketEnv(
            flight_phase=flight_phase,
            num_parallel_envs=num_parallel_envs
        )
        
        # Add vectorized methods to agent
        self._add_vectorized_agent_methods()
        
    def _add_vectorized_agent_methods(self):
        """Add vectorized methods to the agent for parallel processing"""
        
        # Extract relevant params from agent to avoid JIT compilation issues
        actor_params = self.agent.actor_params
        actor = self.agent.actor
        action_dim = self.agent.action_dim
        max_std = self.agent.max_std
        
        @jax.jit
        def _select_actions_vmap_internal(states, rng_key, actor_params, max_std):
            batch_size = states.shape[0]
            keys = jax.random.split(rng_key, batch_size)
            noise = jax.vmap(lambda key: jax.random.normal(key, (action_dim,)))(keys) * max_std
            def select_action_single(state, noise):
                action_mean, action_std = actor.apply(actor_params, state)
                actions = noise * action_std + action_mean
                return actions
            return jax.vmap(select_action_single)(states, noise)
        
        def select_actions_vmap(states): # Inference function
            rng_key = self.agent.get_subkey() # Get a fresh RNG key each time
            return _select_actions_vmap_internal(states, rng_key, actor_params, max_std)
        
        # Extract relevant params from agent to avoid JIT compilation issues
        critic_params = self.agent.critic_params
        critic_target_params = self.agent.critic_target_params
        temperature = self.agent.temperature
        calculate_td_error_lambda = self.agent.calculate_td_error_lambda
        
        @jax.jit
        def _calculate_td_error_vmap_internal(states, actions, rewards, next_states, dones, 
                                             rng_key, actor_params, critic_params, 
                                             critic_target_params, temperature, max_std):
            batch_size = states.shape[0]
            noise = jax.random.normal(rng_key, (batch_size, action_dim)) * max_std
            def process_next_state(next_state, noise):
                next_action_mean, next_action_std = actor.apply(actor_params, next_state)
                next_action = noise * next_action_std + next_action_mean
                next_log_policy = gaussian_likelihood(next_action, next_action_mean, next_action_std)
                return next_action, next_log_policy
            next_actions, next_log_policies = jax.vmap(process_next_state)(next_states, noise)
            def td_error_for_transition(state, action, reward, next_state, done, next_action, next_log_policy):
                return calculate_td_error_lambda(
                    states=state,
                    actions=action,
                    rewards=reward,
                    next_states=next_state,
                    dones=done,
                    temperature=temperature,
                    critic_params=critic_params,
                    critic_target_params=critic_target_params,
                    next_actions=next_action,
                    next_log_policy=next_log_policy
                )
            td_errors = jax.vmap(td_error_for_transition)(states, actions, rewards, next_states, dones, next_actions, next_log_policies)
            return td_errors
        
        # Wrapper for td error calculation
        def calculate_td_error_vmap(states, actions, rewards, next_states, dones):
            rng_key = self.agent.get_subkey()
            return _calculate_td_error_vmap_internal(
                states, actions, rewards, next_states, dones,
                rng_key, actor_params, critic_params, critic_target_params,
                temperature, max_std)
            
        # Add the methods to the agent instance
        self.agent.select_actions_vmap = select_actions_vmap
        self.agent.calculate_td_error_vmap = calculate_td_error_vmap
        
    def train(self):
        """Parallel training implementation with vectorized operations"""
        self.fill_replay_buffer()
        
        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")
        self.critic_warm_up()
        steps_since_last_update = 0
        
        for episode in pbar:
            # Reset all environments in parallel
            states = self.parallel_env.reset()
            
            dones = jnp.zeros(self.num_parallel_envs, dtype=bool)
            truncateds = jnp.zeros(self.num_parallel_envs, dtype=bool)
            total_rewards = jnp.zeros(self.num_parallel_envs)
            
            # Run until all environments are done
            while not jnp.all(jnp.logical_or(dones, truncateds)):
                steps_since_last_update += 1
                
                # Select actions for all active environments using vectorized actor
                active_mask = jnp.logical_not(jnp.logical_or(dones, truncateds))
                active_states = states[active_mask]
                
                if active_states.shape[0] > 0:
                    # Get batch actions using vectorized actor
                    batch_actions = self.agent.select_actions_vmap(active_states)
                    
                    # Step all environments in parallel
                    next_states, rewards, new_dones, new_truncateds, infos = self.parallel_env.step(batch_actions)
                    
                    # Update states and rewards
                    states = states.at[active_mask].set(next_states)
                    total_rewards = total_rewards.at[active_mask].set(total_rewards[active_mask] + rewards)
                    dones = dones.at[active_mask].set(new_dones)
                    truncateds = truncateds.at[active_mask].set(new_truncateds)
                    
                    # Calculate TD errors for all transitions at once using vectorized operation
                    td_errors = self.agent.calculate_td_error_vmap(
                        states[active_mask],
                        batch_actions,
                        rewards,
                        next_states,
                        jnp.logical_or(new_dones, new_truncateds)
                    )
                    
                    # Add to buffer for each active env
                    for i, active in enumerate(active_mask):
                        if active:
                            state = states[i]
                            action = batch_actions[i]
                            reward = rewards[i]
                            next_state = next_states[i]
                            done_or_truncated = new_dones[i] or new_truncateds[i]
                            
                            # Add to buffer
                            self.agent.buffer.add(
                                state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                done=done_or_truncated,
                                td_error=td_errors[i]
                            )
                    
                    # Update the agent
                    if steps_since_last_update % self.update_agent_every_n_steps == 0:
                        self.agent.update()
                        steps_since_last_update = 0
            
            # After all environments finish, update episode stats
            mean_reward = jnp.mean(total_rewards)
            self.epoch_rewards.append(float(mean_reward))
            self.agent.update_episode()
            
            # Log to tensorboard (less frequently)
            if episode % 10 == 0:
                self.agent.writer.add_scalar('Rewards/Reward-per-episode', np.array(mean_reward), episode)
            
            # Update progress bar
            pbar.set_description(f"Training Progress - Episode: {episode}, Mean Reward: {mean_reward:.4e}")
            
            # Save periodically
            if episode % self.save_interval == 0:
                self.save_all()
                
        # Final save and cleanup
        self.save_all()
        self.parallel_env.close()
        print("Training complete.")

class RocketTrainer_SAC_Parallel:
    def __init__(self,
                 flight_phase: str,
                 save_interval: int = 10,
                 load_from: str = 'None',
                 load_buffer_bool: bool = False,
                 pre_train_critic_bool: bool = False,
                 num_parallel_envs: int = 4):
        self.flight_phase = flight_phase
        self.env = ParallelRocketEnv(flight_phase=flight_phase, num_parallel_envs=num_parallel_envs)
        self.num_parallel_envs = num_parallel_envs

        if flight_phase == 'subsonic':
            self.agent_config = config_subsonic
        elif flight_phase == 'supersonic':
            self.agent_config = config_supersonic
        elif flight_phase == 'flip_over_boostbackburn':
            self.agent_config = config_flip_over_boostbackburn
        elif flight_phase == 'ballistic_arc_descent':
            self.agent_config = config_ballistic_arc_descent

        if load_from == 'pso':
            self.load_agent_from_pso()
        elif load_from == 'rl':
            self.load_agent_from_rl()
        elif load_from == 'supervisory':
            self.load_agent_from_supervisory()
        else:
            self.agent = Agent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase=self.flight_phase,
                **self.agent_config['sac'])
            
        if pre_train_critic_bool:
            self.pre_train_critic()

        self.trainer = TrainerSAC_Parallel(
            env=self.env,
            agent=self.agent,
            flight_phase=self.flight_phase,
            num_episodes=self.agent_config['num_episodes'],
            save_interval=save_interval,
            critic_warm_up_steps=self.agent_config['critic_warm_up_steps'],
            load_buffer_from_experiences_bool=load_buffer_bool,
            update_agent_every_n_steps=self.agent_config['update_agent_every_n_steps'],
            num_parallel_envs=num_parallel_envs
        )
        
        self.save_interval = save_interval

    def __call__(self):
        self.trainer.train()

    def load_agent_from_pso(self):
        actor_params, hidden_dim, number_of_hidden_layers = load_pso_actor(self.flight_phase)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = number_of_hidden_layers

        self.agent = Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            flight_phase=self.flight_phase,
            **self.agent_config['sac'])
        self.agent.actor_params = actor_params  

    def load_agent_from_rl(self):
        self.agent = load_sac(f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/soft-actor-critic.pkl')

    def load_agent_from_supervisory(self):
        actor, actor_params, hidden_dim, hidden_layers = load_supervisory_actor(self.flight_phase)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = hidden_layers
        self.agent = Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            flight_phase=self.flight_phase,
            **self.agent_config['sac'])
        actor_params_clean = {'params': actor_params}
        self.agent.actor_params = actor_params_clean        
    
    def pre_train_critic(self):
        critic_params_learner = pre_train_critic_from_pso_experiences(
            flight_phase=self.flight_phase,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim_critic=self.agent_config['sac']['hidden_dim_critic'],
            number_of_hidden_layers_critic=self.agent_config['sac']['number_of_hidden_layers_critic'],
            gamma=self.agent_config['sac']['gamma'],
            tau=self.agent_config['sac']['tau'],
            critic_learning_rate=self.agent_config['pre_train_critic_learning_rate'],
            batch_size=self.agent_config['pre_train_critic_batch_size']
        )
        self.agent.critic_params, self.agent.critic_target_params, self.agent.critic_opt_state = critic_params_learner() 