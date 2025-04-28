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
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.critic_pre_train.pre_train_critic import pre_train_critic_from_pso_experiences
from src.envs.supervisory.agent_load_supervisory import load_supervisory_actor
from src.agents.functions.soft_actor_critic_functions import gaussian_likelihood
from src.envs.universal_physics_plotter import universal_physics_plotter

class TrainerSAC_Vectorized(TrainerSAC):
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
        # Add vectorized methods to agent
        self._add_vectorized_agent_methods()
        self._warmup_jax_operations()
        
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
                actions = noise * jnp.squeeze(action_std) + jnp.squeeze(action_mean)
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
            keys = jax.random.split(rng_key, batch_size)
            noise = jax.vmap(lambda key: jax.random.normal(key, (action_dim,)))(keys) * max_std
            def process_next_state(next_state, noise):
                next_action_mean, next_action_std = actor.apply(actor_params, next_state)
                next_action = noise * jnp.squeeze(next_action_std) + jnp.squeeze(next_action_mean)
                next_log_policy = gaussian_likelihood(next_action, next_action_mean, next_action_std)
                return next_action, next_log_policy
            next_actions, next_log_policies = jax.vmap(process_next_state)(next_states, noise)
            def td_error_for_transition(state, action, reward, next_state, done, next_action, next_log_policy):
                return calculate_td_error_lambda(
                    states=jnp.expand_dims(state, axis=0),
                    actions=jnp.expand_dims(action, axis=0),
                    rewards=jnp.expand_dims(reward, axis=0),
                    next_states=jnp.expand_dims(next_state, axis=0),
                    dones=jnp.expand_dims(done, axis=0),
                    temperature=temperature,
                    critic_params=critic_params,
                    critic_target_params=critic_target_params,
                    next_actions=jnp.expand_dims(next_action, axis=0),
                    next_log_policy=jnp.expand_dims(next_log_policy, axis=0)
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
        """Vectorized training implementation with parallel environments"""
        self.fill_replay_buffer()
        
        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")
        self.critic_warm_up()
        steps_since_last_update = 0
        total_num_steps = 0
        for episode in pbar:
            # Initialize multiple parallel environments
            states = []
            for _ in range(self.num_parallel_envs):
                states.append(self.env.reset())
            states = jnp.array(states)
            
            dones = jnp.zeros(self.num_parallel_envs, dtype=bool)
            truncateds = jnp.zeros(self.num_parallel_envs, dtype=bool)
            total_rewards = jnp.zeros(self.num_parallel_envs)
            num_steps = jnp.zeros(self.num_parallel_envs, dtype=int)
            
            # Run parallel environments until all are done
            while not jnp.all(jnp.logical_or(dones, truncateds)):
                steps_since_last_update += 1
                
                # Select actions for all active environments
                active_mask = jnp.logical_not(jnp.logical_or(dones, truncateds))
                active_states = states[active_mask]
                
                if active_states.shape[0] > 0:  # Only process if we have active environments
                    # Get batch actions using vectorized actor
                    batch_actions = self.agent.select_actions_vmap(active_states)
                    
                    # Step each active environment (can't easily vectorize this part due to environment API)
                    next_states_list = []
                    rewards_list = []
                    dones_list = []
                    truncateds_list = []
                    
                    # Process each active environment
                    env_idx = 0
                    for i in range(self.num_parallel_envs):
                        if active_mask[i]:
                            action = batch_actions[env_idx]
                            next_state, reward, done, truncated, _ = self.env.step(action)
                            
                            next_states_list.append(next_state)
                            rewards_list.append(reward)
                            dones_list.append(done)
                            truncateds_list.append(truncated)
                            
                            # Update state/reward for this environment
                            states = states.at[i].set(next_state)
                            total_rewards = total_rewards.at[i].set(total_rewards[i] + reward)
                            num_steps = num_steps.at[i].set(num_steps[i] + 1)
                            
                            # Update done/truncated flags
                            dones = dones.at[i].set(done)
                            truncateds = truncateds.at[i].set(truncated)
                            
                            # Add to buffer for each active env
                            state = jnp.array(states[i])
                            action = jnp.array(action)
                            reward = jnp.array(reward)
                            next_state = jnp.array(next_state)
                            done_or_truncated = done or truncated
                            done_jnp = jnp.array(done_or_truncated)
                            
                            if action.ndim == 0:
                                action = jnp.expand_dims(action, axis=0)
                                
                            # Calculate TD error for this transition
                            td_error = self.calculate_td_error(
                                jnp.expand_dims(state, axis=0),
                                jnp.expand_dims(action, axis=0),
                                jnp.expand_dims(reward, axis=0),
                                jnp.expand_dims(next_state, axis=0),
                                jnp.expand_dims(done_jnp, axis=0)
                            )
                            
                            # Add to buffer
                            self.agent.buffer.add(
                                state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                done=done_jnp,
                                td_error=jnp.squeeze(td_error)
                            )
                            
                            env_idx += 1
                            total_num_steps += 1
                            
                    # Update the agent
                    if steps_since_last_update % self.update_agent_every_n_steps == 0:
                        self.agent.update()
                        steps_since_last_update = 0
            
            # After all environments finish, update episode stats
            mean_reward = jnp.mean(total_rewards)
            mean_steps = jnp.mean(num_steps)
            self.epoch_rewards.append(float(mean_reward))
            self.agent.update_episode()
            
            # Log to tensorboard (less frequently)
            if episode % 10 == 0:  # Only log every 10 episodes
                self.agent.writer.add_scalar('Rewards/Reward-per-episode', np.array(mean_reward), episode)
                self.agent.writer.add_scalar('Rewards/Episode-steps', np.array(mean_steps), episode)
            
            # Update progress bar
            pbar.set_description(f"Training Progress - Episode: {episode}, Mean Reward: {mean_reward:.4e}, Mean Steps: {mean_steps:.1f}")
            
            # Save periodically
            if episode % self.save_interval == 0:
                self.save_all()
                # Plotting
                universal_physics_plotter(self.env,
                                  self.agent,
                                  self.agent.save_path,
                                  flight_phase = self.env.flight_phase,
                                  type = 'rl')
                
        # Final save
        self.save_all()
        print("Training complete.")

    def _warmup_jax_operations(self):
        """Precompile JAX operations to avoid compilation during actual runs"""
        try:
            # Create dummy data with correct dimensions
            dummy_states = jnp.zeros((self.num_parallel_envs, self.env.state_dim))
            dummy_actions = jnp.zeros((self.num_parallel_envs, self.env.action_dim))
            dummy_rewards = jnp.zeros((self.num_parallel_envs,))
            dummy_next_states = jnp.zeros((self.num_parallel_envs, self.env.state_dim))
            dummy_dones = jnp.zeros((self.num_parallel_envs,), dtype=bool)
            
            # Warm up action selection
            _ = self.agent.select_actions_vmap(dummy_states)
        except Exception as e:
            print(f"Warning: JAX precompilation failed with error (select actions): {e}")
        try:
            # Warm up TD error calculation
            _ = self.agent.calculate_td_error_vmap(
                dummy_states, 
                dummy_actions, 
                dummy_rewards, 
                dummy_next_states, 
                dummy_dones
            )
            
            print("JAX operations precompiled successfully")
        except Exception as e:
            print(f"Warning: JAX precompilation failed with error (calculate td error): {e}")

class RocketTrainer_SAC_Vectorized:
    def __init__(self,
                 flight_phase: str,
                 save_interval: int = 10,
                 load_from: str = 'None',
                 load_buffer_bool: bool = False,
                 pre_train_critic_bool: bool = False,
                 num_parallel_envs: int = 4):
        self.flight_phase = flight_phase
        self.env = env(flight_phase=flight_phase)
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

        self.trainer = TrainerSAC_Vectorized(
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

    def test_performance(self, num_test_episodes=5, buffer_fill_size=10000):
        print(f"Testing with {buffer_fill_size}...")
        self.agent.reset()
        # Time sequential
        print("Running sequential implementation...")
        seq_start = time.time()
        # Run multiple episodes for more accurate timing
        pbar = tqdm(range(num_test_episodes), desc="Sequential implementation progress")
        for _ in pbar:
            state = self.env.reset()
            for i in range(buffer_fill_size // num_test_episodes):
                action = self.agent.select_actions(jnp.expand_dims(state, 0))
                next_state, reward, done, truncated, _ = self.env.step(action)
                # Calculate TD error and add to buffer (but don't store to save memory)
                td_error = self.agent.calculate_td_error(
                    jnp.expand_dims(state, 0),
                    jnp.expand_dims(action, 0),
                    jnp.expand_dims(jnp.array(reward), 0),
                    jnp.expand_dims(next_state, 0),
                    jnp.expand_dims(jnp.array(done), 0)
                )
                state = next_state
                if done or truncated:
                    state = self.env.reset()
                if i % 100 == 0:
                    jax.block_until_ready(td_error)
        jax.block_until_ready(jnp.array(0))
        seq_time = time.time() - seq_start
        
        # Reset for vectorized test
        self.agent.reset()
        print("Running vectorized implementation...")
        vec_start = time.time()
        steps_per_batch = buffer_fill_size // (num_test_episodes * self.num_parallel_envs)
        pbar = tqdm(range(num_test_episodes), desc="Vectorized implementation progress")
        for _ in pbar:
            states = jnp.stack([self.env.reset() for _ in range(self.num_parallel_envs)])
            for i in range(steps_per_batch):
                actions = self.agent.select_actions_vmap(states)
                next_states = []
                rewards = []
                dones = []
                for j in range(self.num_parallel_envs):
                    next_state, reward, done, truncated, _ = self.env.step(actions[j])
                    next_states.append(next_state)
                    rewards.append(reward)
                    dones.append(done or truncated)
                next_states_array = jnp.stack(next_states)
                rewards_array = jnp.array(rewards)
                dones_array = jnp.array(dones)
                td_errors = self.agent.calculate_td_error_vmap(
                    states, actions, rewards_array, next_states_array, dones_array
                )
                states = next_states_array
                for j in range(self.num_parallel_envs):
                    if dones[j]:
                        states = states.at[j].set(self.env.reset())
                if i % 100 == 0:
                    jax.block_until_ready(td_errors)
        jax.block_until_ready(jnp.array(0))
        vec_time = time.time() - vec_start
        
        # Calculate speedup
        speedup = seq_time / vec_time
        
        print(f"\n===== PERFORMANCE COMPARISON ({self.num_parallel_envs} parallel environments) =====")
        print(f"Sequential implementation time: {seq_time:.2f} seconds")
        print(f"Vectorized implementation time: {vec_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Efficiency: {speedup/self.num_parallel_envs*100:.1f}% (100% would be linear scaling)")
        print("===========================================================\n")
        
        return {
            "sequential_time": seq_time,
            "vectorized_time": vec_time,
            "speedup": speedup,
            "efficiency": speedup/self.num_parallel_envs*100,
            "num_parallel_envs": self.num_parallel_envs
        } 

    def test_pure_parallel_performance(self, num_test_episodes=5, buffer_fill_size=10000):
        """Test performance with minimal compiler optimizations to measure pure parallelism gains"""
        print(f"Testing pure parallel performance with {buffer_fill_size} steps...")
        self.agent.reset()
        # Disable JAX optimizations for baseline test; store original flags
        jit_disable = jax.config.jax_disable_jit
        
        try:
            # Time sequential with JIT disabled to measure raw performance
            jax.config.update('jax_disable_jit', True)
            print("Running sequential implementation (no compiler optimizations)...")
            seq_start = time.time()
            pbar = tqdm(range(num_test_episodes), desc="Sequential implementation progress")
            for _ in pbar:
                state = self.env.reset()
                for i in range(buffer_fill_size // num_test_episodes):
                    # Process states one by one
                    actor_output = self.agent.actor.apply(self.agent.actor_params, jnp.array(state))
                    action_mean, action_std = actor_output
                    # Manually generate random actions without JIT
                    key = self.agent.get_subkey()
                    noise = jax.random.normal(key, (self.env.action_dim,))
                    action = noise * action_std + action_mean
                    # Step environment
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    state = next_state
                    if done or truncated:
                        state = self.env.reset()
                        
            # Ensure all operations are complete
            jax.block_until_ready(jnp.array(0))
            seq_time = time.time() - seq_start
            
            # Now test parallel with JIT still disabled
            print("Running parallel implementation (no compiler optimizations)...")
            vec_start = time.time()
            steps_per_batch = buffer_fill_size // (num_test_episodes * self.num_parallel_envs)
            pbar = tqdm(range(num_test_episodes), desc="Parallel implementation progress")
            for _ in pbar:
                all_states = [self.env.reset() for _ in range(self.num_parallel_envs)]
                for i in range(steps_per_batch):
                    all_next_states = []
                    all_rewards = []
                    all_dones = []
                    all_actions = []
                    # Process each state in "parallel" (but without vmap/JIT)
                    for j in range(self.num_parallel_envs):
                        state = all_states[j]
                        # Get action for this state
                        actor_output = self.agent.actor.apply(self.agent.actor_params, jnp.array(state))
                        action_mean, action_std = actor_output
                        # Manually generate random action
                        key = self.agent.get_subkey()
                        noise = jax.random.normal(key, (self.env.action_dim,))
                        action = noise * action_std + action_mean
                        all_actions.append(action)
                        # Step environment
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        all_next_states.append(next_state)
                        all_rewards.append(reward)
                        all_dones.append(done or truncated)
                    # Update states
                    for j in range(self.num_parallel_envs):
                        if all_dones[j]:
                            all_states[j] = self.env.reset()
                        else:
                            all_states[j] = all_next_states[j]
                            
            # Ensure all operations are complete  
            jax.block_until_ready(jnp.array(0))
            vec_time = time.time() - vec_start
            speedup = seq_time / vec_time
            
            print(f"\n===== PURE PARALLEL PERFORMANCE ({self.num_parallel_envs} parallel environments) =====")
            print(f"Sequential implementation time (no JIT): {seq_time:.2f} seconds")
            print(f"Parallel implementation time (no JIT): {vec_time:.2f} seconds")
            print(f"Speedup from pure parallelism: {speedup:.2f}x")
            print(f"Efficiency: {speedup/self.num_parallel_envs*100:.1f}% (100% would be linear scaling)")
            print("===========================================================\n")
            
            return {
                "sequential_time_no_jit": seq_time,
                "parallel_time_no_jit": vec_time,
                "pure_parallelism_speedup": speedup,
                "pure_parallelism_efficiency": speedup/self.num_parallel_envs*100,
                "num_parallel_envs": self.num_parallel_envs
            }
            
        finally:
            # Restore original flags
            jax.config.update('jax_disable_jit', jit_disable) 

    def run_benchmarks(self, num_test_episodes=5, buffer_fill_size=10000):
        """Run comprehensive benchmarks to measure different factors in performance gains"""
        
        
        
        # 2. Run without JIT to measure pure parallelism
        print("\n=== RUNNING PURE PARALLELISM BENCHMARK (NO COMPILER OPTIMIZATIONS) ===")
        pure_parallel_results = self.test_pure_parallel_performance(num_test_episodes, buffer_fill_size)

        # 1. Run the original benchmark to measure full optimization
        print("\n=== RUNNING FULL OPTIMIZATION BENCHMARK ===")
        full_opt_results = self.test_performance(num_test_episodes, buffer_fill_size)
        
        # 3. Calculate how much of the speedup comes from compiler optimizations
        optimization_factor = full_opt_results["speedup"] / pure_parallel_results["pure_parallelism_speedup"]
        
        # Print summary
        print("\n======= BENCHMARK SUMMARY =======")
        print(f"Number of parallel environments: {self.num_parallel_envs}")
        print(f"Full optimization speedup: {full_opt_results['speedup']:.2f}x")
        print(f"Pure parallelism speedup: {pure_parallel_results['pure_parallelism_speedup']:.2f}x")
        print(f"Compiler optimization factor: {optimization_factor:.2f}x")
        print(f"Percentage of speedup from compiler optimization: {(optimization_factor-1)/optimization_factor*100:.1f}%")
        print(f"Percentage of speedup from parallelism: {1/optimization_factor*100:.1f}%")
        print("==================================\n")
        
        return {
            "full_optimization": full_opt_results,
            "pure_parallelism": pure_parallel_results,
            "optimization_factor": optimization_factor,
            "optimization_percentage": (optimization_factor-1)/optimization_factor*100,
            "parallelism_percentage": 1/optimization_factor*100
        } 
    


def test_vectorized_performance():
    print("Testing vectorized implementation with different numbers of parallel environments")
    save_folder = "results/parallelism_benchmarks/vectorized_performance"
    os.makedirs(save_folder, exist_ok=True)
    flight_phase = "subsonic"
    buffer_fill_size = 2000
    parallel_envs_list = [1, 2, 4]
    
    # Results storage
    results = {
        "num_parallel_envs": [],
        "sequential_time": [],
        "vectorized_time": [],
        "speedup": [],
        "efficiency": []
    }
    
    for num_envs in parallel_envs_list:
        print(f"\nTesting with {num_envs} parallel environments...")
        trainer = RocketTrainer_SAC_Vectorized(
            flight_phase=flight_phase,
            save_interval=1000,  # Don't save during test
            num_parallel_envs=num_envs
        )
        
        perf_results = trainer.test_performance(
            num_test_episodes=3,
            buffer_fill_size=buffer_fill_size
        )
        for key in results.keys():
            results[key].append(perf_results[key])
    
    
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    plt.suptitle(f'Performance Scaling with Parallelism', fontsize=32)
    ax1 = plt.subplot(gs[0])
    ax1.plot(results["num_parallel_envs"], results["speedup"], 'o-', linewidth=4, color='blue')
    ax1.plot(results["num_parallel_envs"], results["num_parallel_envs"], 'k--', label="Linear speedup", linewidth=4)
    ax1.set_ylabel("Speedup Factor [-]", fontsize=20)
    ax1.set_title("Performance Scaling with Parallelism", fontsize=22)
    ax1.grid(True)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(results["num_parallel_envs"], results["efficiency"], 'o-', linewidth=4, color='blue')
    ax2.axhline(y=100, linestyle='--', label="100% efficiency", linewidth=4, color='black')
    ax2.set_xlabel("Number of Parallel Environments", fontsize=20)
    ax2.set_ylabel("Efficiency [%]", fontsize=20)
    ax2.grid(True)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    plt.savefig(f"{save_folder}/vectorized_scaling.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(20, 15))
    x = np.arange(len(results["num_parallel_envs"]))
    width = 0.35
    plt.bar(x - width/2, results["sequential_time"], width, label='Sequential')
    plt.bar(x + width/2, results["vectorized_time"], width, label='Vectorized')
    plt.xlabel('Number of Parallel Environments', fontsize=20)
    plt.ylabel('Execution Time (seconds)', fontsize=20)
    plt.title('Execution Time Comparison', fontsize=22)
    plt.xticks(x, results["num_parallel_envs"], fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{save_folder}/execution_times.png", dpi=300)
    plt.close()
    
    # Print final summary
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"{'Num Envs':<10} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 40)
    for i, n in enumerate(results["num_parallel_envs"]):
        print(f"{n:<10} {results['vectorized_time'][i]:<10.2f} {results['speedup'][i]:<10.2f} {results['efficiency'][i]:<10.1f}%")
    print("==============================\n")
    
    # Save results to file
    with open(f"{save_folder}/vectorized_results.txt", "w") as f:
        f.write("===== PERFORMANCE SUMMARY =====\n")
        f.write(f"{'Num Envs':<10} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<10}\n")
        f.write("-" * 40 + "\n")
        for i, n in enumerate(results["num_parallel_envs"]):
            f.write(f"{n:<10} {results['vectorized_time'][i]:<10.2f} {results['speedup'][i]:<10.2f} {results['efficiency'][i]:<10.1f}%\n")
        f.write("==============================\n")