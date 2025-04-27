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
                 update_agent_every_n_steps: int = 10):
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
        plt.plot(self.epoch_rewards, label="Episode Rewards", alpha=0.5)
        plt.plot(moving_avg, 
                label=f"{window_size}-Episode Moving Average",
                linewidth=2)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Rewards Over Training")
        plt.legend()
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
        all_experiences_unpacked = False
        with env.begin() as txn:
            cursor = txn.cursor()
            while len(self.agent.buffer) < self.buffer_size or all_experiences_unpacked:
                for _, value in cursor:
                    experience = pickle.loads(value)  # Unpack experience
                    state, action, reward, next_state, _ = experience
                    state_jnp = jnp.array(state)
                    action_jnp = jnp.array(action.detach().numpy())
                    reward_jnp = jnp.array(reward)
                    next_state_jnp = jnp.array(next_state)
                    done = 0
                    done_jnp = jnp.array(done)

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
                all_experiences_unpacked = True

    def fill_replay_buffer(self):
        """
        Fill the replay buffer with initial experience using random actions.
        """
        if self.load_buffer_from_experiences_bool == True:
            self.add_experiences_to_buffer()
        while len(self.agent.buffer) < self.buffer_size:        
            state = self.env.reset()
            done = False
            while not done:
                # Sample random action and ensure it's a jax array
                action = self.agent.select_actions(jnp.expand_dims(state, 0)) 
                action = jnp.array(action)
                if action.ndim == 0:
                    action = jnp.expand_dims(action, 0)
                
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
                state = next_state

    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        pass # Placeholder for child classes

    def critic_warm_up(self):
        pass # Placeholder for child classes

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
            self.agent.writer.flush()
            # Plot the rewards and losses
            if episode % self.save_interval == 0:
                self.save_all()

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
                 load_buffer_from_experiences_bool : bool = False,
                 update_agent_every_n_steps: int = 10):
        """
        Initialize the trainer.
        
        Args:
            env: The environment in which the agent operates.
            agent: The agent being trained.
            num_episodes: Number of training episodes
            buffer_size: Replay buffer size [int]
        """
        super(TrainerSAC, self).__init__(env, agent, load_buffer_from_experiences_bool, flight_phase, num_episodes, save_interval, critic_warm_up_steps, update_agent_every_n_steps)

    # Could become jittable.
    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        return self.agent.calculate_td_error(states, actions, rewards, next_states, dones)
    
    def critic_warm_up(self):
        pbar = tqdm(range(1, self.critic_warm_up_steps + 1), desc="Critic Warm Up Progress")
        for _ in pbar:
            critic_warm_up_loss = self.agent.critic_warm_up_step()
            pbar.set_description(f"Critic Warm Up Progress - Loss: {critic_warm_up_loss:.4e}")

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