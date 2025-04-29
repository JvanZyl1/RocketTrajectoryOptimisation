import os
import torch
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import lmdb
import pickle

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def moving_average(var, window_size=5):
    window_size = min(5, len(var))
    moving_avg = []
    for i in range(len(var)):
        if i < window_size:
            moving_avg.append(sum(var[:i+1]) / (i+1))
        else:
            moving_avg.append(sum(var[i-window_size+1:i+1]) / window_size)
    return moving_avg


# Parent Trainer class
class TrainerSkeleton:
    def __init__(self,
                 env,
                 agent,
                 load_buffer_from_experiences_bool : bool,
                 flight_phase : str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 critic_warm_up_early_stopping_loss: float = 0.0,
                 update_agent_every_n_steps: int = 10,
                 priority_update_interval: int = 5):
        self.env = env
        self.agent = agent
        self.gamma = agent.gamma
        self.num_episodes = num_episodes
        self.buffer_size = agent.buffer.buffer_size
        self.save_interval = save_interval
        self.dt = self.env.dt
        self.critic_warm_up_steps = critic_warm_up_steps
        self.epoch_rewards = []
        self.flight_phase = flight_phase
        self.load_buffer_from_experiences_bool = load_buffer_from_experiences_bool
        self.update_agent_every_n_steps = update_agent_every_n_steps
        self.critic_warm_up_early_stopping_loss = critic_warm_up_early_stopping_loss
        self.priority_update_interval = priority_update_interval

    def plot_rewards(self):
        save_path_rewards = self.agent.save_path + 'rewards.png'
        
        # Calculate moving average
        window_size = min(5, len(self.epoch_rewards))
        moving_avg = []
        for i in range(len(self.epoch_rewards)):
            if i < window_size:
                moving_avg.append(sum(self.epoch_rewards[:i+1]) / (i+1))
            else:
                moving_avg.append(sum(self.epoch_rewards[i-window_size+1:i+1]) / window_size)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_rewards, label="Episode Rewards", alpha=0.5, linewidth=4, color = 'pink', linestyle = '--')
        plt.plot(moving_avg, 
                label=f"{window_size}-Episode Moving Average",
                linewidth=4,
                color = 'blue')
        plt.xlabel("Episodes", fontsize = 20)
        plt.ylabel("Total Reward", fontsize = 20)
        plt.title("Rewards Over Training", fontsize = 22)
        plt.legend(fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.grid()
        plt.savefig(save_path_rewards, format='png', dpi=300)
        plt.close()

    def save_all(self):
        self.plot_rewards()
        self.agent.plotter()
        self.agent.save()
        if hasattr(self, 'test_env'):
            self.test_env()

    def add_experiences_to_buffer(self):
        folder_path=f"data/experience_buffer/{self.flight_phase}/experience_buffer.lmdb"
        env = lmdb.open(folder_path, readonly=True, lock=False)  # Open the folder, not the .mdb file
        
        # Count non-zero experiences in buffer and convert to Python int
        non_zero_experiences = int(jnp.sum(jnp.any(self.agent.buffer.buffer != 0, axis=1)))
        remaining_experiences = self.buffer_size - non_zero_experiences
        
        # Batch size for processing
        batch_size = 1000
        experiences_batch = []
        
        pbar = tqdm(total=remaining_experiences, desc="Loading experiences from file")
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                if non_zero_experiences >= self.buffer_size:
                    break
                    
                experience = pickle.loads(value)  # Unpack experience
                experiences_batch.append(experience)
                
                if len(experiences_batch) >= batch_size or non_zero_experiences + len(experiences_batch) >= self.buffer_size:
                    # Process batch
                    states = jnp.array([exp[0] for exp in experiences_batch])
                    actions = jnp.array([exp[1].detach().numpy() for exp in experiences_batch])
                    rewards = jnp.array([exp[2] for exp in experiences_batch])
                    next_states = jnp.array([exp[3] for exp in experiences_batch])
                    dones = jnp.zeros(len(experiences_batch))
                    
                    # Calculate TD errors in batch using vectorized method
                    td_errors = self.agent.calculate_td_error_vmap(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        dones=dones
                    )
                    
                    # Add experiences to buffer in batch
                    for i in range(len(experiences_batch)):
                        self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i],
                            next_state=next_states[i],
                            done=dones[i],
                            td_error=td_errors[i]
                        )
                        non_zero_experiences += 1
                        pbar.update(1)
                    
                    # Update priorities in batch
                    indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                                      self.agent.buffer.position)
                    self.agent.buffer.update_priorities(indices, td_errors)
                    
                    experiences_batch = []
        
        # Process any remaining experiences
        if experiences_batch:
            states = jnp.array([exp[0] for exp in experiences_batch])
            actions = jnp.array([exp[1].detach().numpy() for exp in experiences_batch])
            rewards = jnp.array([exp[2] for exp in experiences_batch])
            next_states = jnp.array([exp[3] for exp in experiences_batch])
            dones = jnp.zeros(len(experiences_batch))
            
            # Calculate TD errors in batch using vectorized method
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            for i in range(len(experiences_batch)):
                self.agent.buffer.add(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=dones[i],
                    td_error=td_errors[i]
                )
                non_zero_experiences += 1
                pbar.update(1)
            
            indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                               self.agent.buffer.position)
            self.agent.buffer.update_priorities(indices, td_errors)
        
        # Save buffer after filling
        buffer_save_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/buffer_after_loading.pkl'
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)

        # Create a dictionary with all buffer components
        buffer_state = {
            'buffer': self.agent.buffer.buffer,
            'priorities': self.agent.buffer.priorities,
            'n_step_buffer': self.agent.buffer.n_step_buffer,
            'position': self.agent.buffer.position,
            'beta': self.agent.buffer.beta
        }

        with open(buffer_save_path, 'wb') as f:
            pickle.dump(buffer_state, f)
        print(f"Saved complete buffer state to {buffer_save_path}")

    def fill_replay_buffer(self):
        """
        Fill the replay buffer with initial experience using random actions.
        """
        if self.load_buffer_from_experiences_bool == True:
            self.add_experiences_to_buffer()
        else:
            self.load_buffer_from_rl()
        
        # Count non-zero experiences in buffer and convert to Python int
        non_zero_experiences = int(jnp.sum(jnp.any(self.agent.buffer.buffer != 0, axis=1)))
        
        # Calculate how many more experiences we need
        remaining_experiences = self.buffer_size - non_zero_experiences
        
        # Batch size for processing
        batch_size = 100
        experiences_batch = []
        
        pbar = tqdm(total=remaining_experiences, desc="Filling replay buffer")
        while non_zero_experiences < self.buffer_size:
            state = self.env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                # Sample random action and ensure it's a jax array
                action = self.agent.select_actions(jnp.expand_dims(state, 0)) 
                action = jnp.array(action)
                if action.ndim == 0:
                    action = jnp.expand_dims(action, 0)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                done_or_truncated = done or truncated
                
                # Add experience to batch and update progress
                experiences_batch.append((state, action, reward, next_state, done_or_truncated))
                non_zero_experiences += 1
                pbar.update(1)
                state = next_state
                
                # Process batch if full or if we've reached buffer size
                if len(experiences_batch) >= batch_size or non_zero_experiences >= self.buffer_size:
                    # Process batch
                    states = jnp.array([exp[0] for exp in experiences_batch])
                    actions = jnp.array([exp[1] for exp in experiences_batch])
                    rewards = jnp.array([exp[2] for exp in experiences_batch])
                    next_states = jnp.array([exp[3] for exp in experiences_batch])
                    dones = jnp.array([exp[4] for exp in experiences_batch])
                    
                    # Ensure proper shapes for concatenation
                    states = jnp.reshape(states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
                    actions = jnp.reshape(actions, (len(experiences_batch), -1))  # Reshape to (batch_size, action_dim)
                    rewards = jnp.reshape(rewards, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
                    next_states = jnp.reshape(next_states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
                    dones = jnp.reshape(dones, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
                    
                    # Calculate TD errors in batch using vectorized method
                    td_errors = self.agent.calculate_td_error_vmap(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        dones=dones
                    )
                    
                    # Add experiences to buffer in batch
                    for i in range(len(experiences_batch)):
                        self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i][0],
                            next_state=next_states[i],
                            done=dones[i][0],
                            td_error=jnp.squeeze(td_errors[i])
                        )
                    
                    # Update priorities in batch
                    indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                                      self.agent.buffer.position)
                    self.agent.buffer.update_priorities(indices, td_errors)
                    
                    experiences_batch = []
                
                if non_zero_experiences >= self.buffer_size:
                    break
        
        # Process any remaining experiences
        if experiences_batch:
            states = jnp.array([exp[0] for exp in experiences_batch])
            actions = jnp.array([exp[1] for exp in experiences_batch])
            rewards = jnp.array([exp[2] for exp in experiences_batch])
            next_states = jnp.array([exp[3] for exp in experiences_batch])
            dones = jnp.array([exp[4] for exp in experiences_batch])
            
            # Ensure proper shapes for concatenation
            states = jnp.reshape(states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
            actions = jnp.reshape(actions, (len(experiences_batch), -1))  # Reshape to (batch_size, action_dim)
            rewards = jnp.reshape(rewards, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
            next_states = jnp.reshape(next_states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
            dones = jnp.reshape(dones, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
            
            # Calculate TD errors in batch using vectorized method
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            for i in range(len(experiences_batch)):
                self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i][0],
                            next_state=next_states[i],
                            done=dones[i][0],
                            td_error=jnp.squeeze(td_errors[i])
                        )
            
            indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                               self.agent.buffer.position)
            self.agent.buffer.update_priorities(indices, td_errors)
        
        self.save_buffer()
    def save_buffer(self):
        # Save buffer after filling
        buffer_save_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/buffer_after_filling.pkl'
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)

        # Create a dictionary with all buffer components
        buffer_state = {
            'buffer': self.agent.buffer.buffer,
            'priorities': self.agent.buffer.priorities,
            'n_step_buffer': self.agent.buffer.n_step_buffer,
            'position': self.agent.buffer.position,
            'beta': self.agent.buffer.beta
        }

        with open(buffer_save_path, 'wb') as f:
            pickle.dump(buffer_state, f)
        print(f"Saved complete buffer state to {buffer_save_path}")

    def load_buffer_from_rl(self):
        buffer_save_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/buffer_after_filling.pkl'
        try:
            with open(buffer_save_path, 'rb') as f:
                buffer_state = pickle.load(f)
                
                # Load all buffer components
                if isinstance(buffer_state, dict):
                    # New format with separate components
                    # Update buffer array using JAX's functional update pattern - vectorized approach
                    buffer_length = len(buffer_state['buffer'])
                    indices = jnp.arange(buffer_length)
                    
                    # Use vmap to vectorize the update operations when possible
                    self.agent.buffer.buffer = self.agent.buffer.buffer.at[indices].set(buffer_state['buffer'][:buffer_length])
                    self.agent.buffer.priorities = self.agent.buffer.priorities.at[indices].set(buffer_state['priorities'][:buffer_length])
                    
                    # For n_step_buffer, we need to update each entry
                    n_step_length = len(buffer_state['n_step_buffer'])
                    n_step_indices = jnp.arange(n_step_length)
                    self.agent.buffer.n_step_buffer = self.agent.buffer.n_step_buffer.at[n_step_indices].set(buffer_state['n_step_buffer'][:n_step_length])
                    
                    # Update scalar values
                    self.agent.buffer.position = int(buffer_length)
                    self.agent.buffer.beta = float(buffer_state['beta'])
                else:
                    # Legacy format - entire buffer object
                    self.agent.buffer = buffer_state
            
            print(f"Loaded buffer from {buffer_save_path}")
            number_of_experiences = len(buffer_state['buffer'])
            print(f"Number of experiences in buffer: {number_of_experiences}")
        except FileNotFoundError:
            print(f"Buffer file not found at {buffer_save_path}. Please ensure the file exists.")

    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        pass # Placeholder for child classes

    def critic_warm_up(self):
        pbar = tqdm(range(1, self.critic_warm_up_steps + 1), desc="Critic Warm Up Progress")
        for _ in pbar:
            critic_warm_up_loss = self.agent.critic_warm_up_step()
            pbar.set_description(f"Critic Warm Up Progress - Loss: {critic_warm_up_loss:.4e}")
            if critic_warm_up_loss < self.critic_warm_up_early_stopping_loss:
                break
        
        # Recalculate TD errors for all experiences in buffer using warmed-up critic
        print("Recalculating TD errors for buffer experiences after critic warmup...")
        
        # Extract non-empty experiences from buffer
        non_empty_mask = jnp.any(self.agent.buffer.buffer != 0, axis=1)
        indices = jnp.where(non_empty_mask)[0]
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch of experiences
            batch_experiences = self.agent.buffer.buffer[batch_indices]
            
            # Extract components
            states = batch_experiences[:, :self.agent.state_dim]
            actions = batch_experiences[:, self.agent.state_dim:self.agent.state_dim + self.agent.action_dim]
            rewards = batch_experiences[:, self.agent.state_dim + self.agent.action_dim]
            next_states = batch_experiences[:, self.agent.state_dim + self.agent.action_dim + 1:
                                           self.agent.state_dim * 2 + self.agent.action_dim + 1]
            dones = batch_experiences[:, self.agent.state_dim * 2 + self.agent.action_dim + 1]
            
            # Calculate new TD errors with warmed-up critic using vmap
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            # Update the TD errors and priorities in the buffer
            # Ensure td_errors has the right shape for updating at the last column
            if td_errors.ndim > 1:
                td_errors = jnp.squeeze(td_errors)
                
            self.agent.buffer.buffer = self.agent.buffer.buffer.at[batch_indices, -1].set(td_errors)
            self.agent.buffer.priorities = self.agent.buffer.priorities.at[batch_indices].set(jnp.abs(td_errors) + 1e-6)

    def update_all_priorities(self):
        """Recalculate TD errors for all experiences in buffer to keep priorities current with the improving critic."""
        print("Recalculating TD errors for all experiences in buffer...")
        
        # Extract non-empty experiences from buffer
        non_empty_mask = jnp.any(self.agent.buffer.buffer != 0, axis=1)
        indices = jnp.where(non_empty_mask)[0]
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch of experiences
            batch_experiences = self.agent.buffer.buffer[batch_indices]
            
            # Extract components
            states = batch_experiences[:, :self.agent.state_dim]
            actions = batch_experiences[:, self.agent.state_dim:self.agent.state_dim + self.agent.action_dim]
            rewards = batch_experiences[:, self.agent.state_dim + self.agent.action_dim]
            next_states = batch_experiences[:, self.agent.state_dim + self.agent.action_dim + 1:
                                           self.agent.state_dim * 2 + self.agent.action_dim + 1]
            dones = batch_experiences[:, self.agent.state_dim * 2 + self.agent.action_dim + 1]
            
            # Calculate new TD errors with current critic using vmap
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            # Ensure td_errors has the right shape for updating
            if td_errors.ndim > 1:
                td_errors = jnp.squeeze(td_errors)
                
            # Update the TD errors in the buffer and priorities
            self.agent.buffer.buffer = self.agent.buffer.buffer.at[batch_indices, -1].set(td_errors)
            self.agent.buffer.priorities = self.agent.buffer.priorities.at[batch_indices].set(jnp.abs(td_errors) + 1e-6)
        
        print("Priority update complete.")

    def train(self):
        """
        Train the agent and log progress.
        """
        # Fill the replay buffer before training
        self.fill_replay_buffer()                       # Randomly fill buffer.

        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")

        self.critic_warm_up()
        steps_since_last_update = 0
        
        total_num_steps = 0
        for episode in pbar:
            state = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            num_steps = 0
            episode_time = 0.0

            while not (done or truncated):
                steps_since_last_update += 1
                # Sample action from the agent, use sample actions function as a stochastic policy
                action = self.agent.select_actions(jnp.expand_dims(state, 0))  # Add batch dimension for input

                # Take a step in the environment
                next_state, reward, done, truncated, _ = self.env.step(action)
                done_or_truncated = done or truncated

                # Add the experience to the replay buffer
                state_jnp = jnp.array(state)
                action_jnp = jnp.array(action)
                reward_jnp = jnp.array(reward)
                next_state_jnp = jnp.array(next_state)
                done_jnp = jnp.array(done_or_truncated)

                if action_jnp.ndim == 0:
                    action_jnp = jnp.expand_dims(action_jnp, axis=0)

                td_error = self.calculate_td_error(
                    jnp.expand_dims(state_jnp, axis=0),
                    jnp.expand_dims(action_jnp, axis=0),
                    jnp.expand_dims(reward_jnp, axis=0),
                    jnp.expand_dims(next_state_jnp, axis=0),
                    jnp.expand_dims(done_jnp, axis=0)
                )

                self.agent.buffer.add(
                    state=state_jnp,
                    action=action_jnp,
                    reward=reward_jnp,
                    next_state=next_state_jnp,
                    done=done_jnp,
                    td_error= jnp.squeeze(td_error)
                )

                # Update the agent
                if steps_since_last_update % self.update_agent_every_n_steps == 0 and steps_since_last_update != 0:
                    self.agent.update()
                    steps_since_last_update = 0

                # Update the state and total reward
                state = next_state_jnp
                total_reward += reward_jnp
                num_steps += 1
                episode_time += self.dt
                total_num_steps += 1
                self.agent.writer.add_scalar('Rewards/Reward-per-step', np.array(reward_jnp), total_num_steps)

                # If done:
                if done_or_truncated:
                    self.agent.update_episode()

            # Log the total reward for the episode
            self.epoch_rewards.append(total_reward)
            self.agent.writer.add_scalar('Rewards/Reward-per-episode', np.array(total_reward), episode)
            self.agent.writer.add_scalar('Rewards/Episode-time', np.array(episode_time), episode)
            pbar.set_description(f"Training Progress - Episode: {episode}, Total Reward: {total_reward:.4e}, Num Steps: {num_steps}:")
            
            # Periodically update all priorities in the buffer to keep them current with the improving critic
            if episode % self.priority_update_interval == 0:
                self.update_all_priorities()
                
            # Plot the rewards and losses
            if episode % self.save_interval == 0:
                self.save_all()
                
            self.agent.writer.flush()

        self.save_all()
        print("Training complete.")

    def plot_final_run(self):
        pass

    def test_env(self):
        pass

#### Soft-Actor Critic ####
class TrainerSAC(TrainerSkeleton):
    """
    Trainer class for the Soft Actor Critic agent.
    """
    def __init__(self,
                 env,
                 agent,
                 flight_phase : str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 critic_warm_up_early_stopping_loss: float = 0.0,
                 load_buffer_from_experiences_bool : bool = False,
                 update_agent_every_n_steps: int = 10,
                 priority_update_interval: int = 5):
        """
        Initialize the trainer.
        
        Args:
            env: The environment in which the agent operates.
            agent: The agent being trained.
            num_episodes: Number of training episodes
            buffer_size: Replay buffer size [int]
        """
        super(TrainerSAC, self).__init__(env, agent, load_buffer_from_experiences_bool, flight_phase, num_episodes, 
                                         save_interval, critic_warm_up_steps, critic_warm_up_early_stopping_loss, 
                                         update_agent_every_n_steps, priority_update_interval)

    # Could become jittable.
    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        return self.agent.calculate_td_error(states, actions, rewards, next_states, dones)
    
    def plot_critic_warmup(self, critic_warmup_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(critic_warmup_losses, label='Critic Warmup Loss', color='blue', linewidth=4)
        plt.xlabel('Step', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.title('Critic Warmup Loss', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(f'results/VanillaSAC/{self.flight_phase}/critic_warmup_loss.png')
        plt.close()
    
    def critic_warm_up(self):
        pbar = tqdm(range(1, self.critic_warm_up_steps + 1), desc="Critic Warm Up Progress")
        critic_warmup_losses = []
        for _ in pbar:
            critic_warm_up_loss = self.agent.critic_warm_up_step()
            pbar.set_description(f"Critic Warm Up Progress - Loss: {critic_warm_up_loss:.4e}")
            critic_warmup_losses.append(critic_warm_up_loss)
            if critic_warm_up_loss < self.critic_warm_up_early_stopping_loss:
                break
        
        # Recalculate TD errors for all experiences in buffer using warmed-up critic
        print("Recalculating TD errors for buffer experiences after critic warmup SAC...")
        
        # Extract non-empty experiences from buffer
        non_empty_mask = jnp.any(self.agent.buffer.buffer != 0, axis=1)
        indices = jnp.where(non_empty_mask)[0]
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch of experiences
            batch_experiences = self.agent.buffer.buffer[batch_indices]
            
            # Extract components
            states = batch_experiences[:, :self.agent.state_dim]
            actions = batch_experiences[:, self.agent.state_dim:self.agent.state_dim + self.agent.action_dim]
            rewards = batch_experiences[:, self.agent.state_dim + self.agent.action_dim]
            next_states = batch_experiences[:, self.agent.state_dim + self.agent.action_dim + 1:
                                           self.agent.state_dim * 2 + self.agent.action_dim + 1]
            dones = batch_experiences[:, self.agent.state_dim * 2 + self.agent.action_dim + 1]
            
            # Calculate new TD errors with warmed-up critic using vmap
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            # Update the TD errors and priorities in the buffer
            # Ensure td_errors has the right shape for updating at the last column
            if td_errors.ndim > 1:
                td_errors = jnp.squeeze(td_errors)
                
            self.agent.buffer.buffer = self.agent.buffer.buffer.at[batch_indices, -1].set(td_errors)
            self.agent.buffer.priorities = self.agent.buffer.priorities.at[batch_indices].set(jnp.abs(td_errors) + 1e-6)
        
        self.save_buffer()
        self.plot_critic_warmup(critic_warmup_losses)

#### Stable Baselines 3 ####
def trainer_StableBaselines3(env,
                             model_name: str):
    
    log_dir = f'data/agent_saves/StableBaselines3/{model_name}/logs'
    model_dir = f'data/agent_saves/StableBaselines3/{model_name}/models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create the model
    model = SAC("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                gradient_steps=-1,
                learning_rate=2e-3,
                buffer_size=100000,
                batch_size=512,
                gamma=0.99,
                policy_kwargs={
                    "net_arch": [256, 256, 256, 256, 256],
                    "clip_mean": 1.0,
                    "activation_fn": torch.nn.Tanh,
                    "use_sde": True,
                    "use_expln": True
                })


    # Create a callback for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=model_dir,
        name_prefix="sac_endo_ascent_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Train the model
    model.learn(total_timesteps=400000,
                log_interval=25,
                callback=checkpoint_callback)

    # Save the final model
    model.save(f"{model_dir}/sac_endo_ascent_final")