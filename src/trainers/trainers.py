import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.agents.functions.plotter import moving_average

# Parent Trainer class
class TrainerSkeleton:
    def __init__(self,
                 env,
                 agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 info: str = ""):
        self.env = env
        self.agent = agent
        self.gamma = agent.gamma
        self.num_episodes = num_episodes
        self.buffer_size = agent.buffer.buffer_size
        self.save_interval = save_interval
        self.info = info

        self.dt = self.env.dt

        self.epoch_rewards = []

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
        # Plot rewards
        self.plot_rewards()
        # Plot agent logs
        self.agent.plotter()
        # Save agent and buffer
        self.agent.save(self.info)
        # If a test_env function is defined, run it
        if hasattr(self, 'test_env'):
            self.test_env()

    def fill_replay_buffer(self):
        """
        Fill the replay buffer with initial experience using random actions.
        """
        while len(self.agent.buffer) < self.buffer_size:
            state = self.env.reset()
            done = False
            while not done:
                # Sample random action and ensure it's a jax array
                action = self.env.action_space.sample()
                action = jnp.array(action)
                if action.ndim == 0:
                    action = jnp.expand_dims(action, 0)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.agent.buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated,
                    td_error=0.0
                )
                state = next_state

    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        pass # Placeholder for child classes

    def train(self):
        """
        Train the agent and log progress.
        """
        # Fill the replay buffer before training
        self.fill_replay_buffer()

        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")
        
        total_num_steps = 0
        for episode in pbar:
            state = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            num_steps = 0
            episode_time = 0.0

            while not (done or truncated):
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
                self.agent.update()

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
            pbar.set_description(f"Training Progress - Episode: {episode}, Total Reward: {total_reward:.2f}, Num Steps: {num_steps}:")
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
                 num_episodes: int,
                 save_interval: int = 10,
                 info: str = ""):
        """
        Initialize the trainer.

        Args:
            env: The environment in which the agent operates.
            agent: The agent being trained.
            num_episodes: Number of training episodes
            buffer_size: Replay buffer size [int]
        """
        super(TrainerSAC, self).__init__(env, agent, num_episodes, save_interval, info)

    # Could become jittable.
    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        return self.agent.calculate_td_error(states, actions, rewards, next_states, dones)
    

#### MARL #####
class Trainer_MARL:
    def __init__(self,
                 env,
                 worker_agent_clone,
                 central_agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 number_of_agents: int = 2,
                 info: str = ""):
        self.env = env
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.number_of_agents = number_of_agents
        self.info = info

        self.epoch_rewards = []

        self.worker_agents = []
        for i in range(number_of_agents):
            self.worker_agents.append(worker_agent_clone)
        self.central_agent = central_agent

        self.epoch_rewards_workers = np.zeros((number_of_agents, num_episodes))

    def plot_rewards(self, episode_no):
        save_path_rewards = self.central_agent.save_path + 'rewards.png'
        save_path_rewards_central = self.central_agent.save_path + 'rewards_central.png'
        save_path_rewards_workers = self.central_agent.save_path + 'rewards_workers.png'
        
        # Convert to list
        window_size = 5
        moving_avg_central = moving_average(self.epoch_rewards, window_size)

        moving_avg_workers = []
        for i in range(self.number_of_agents):
            rewards_of_this_worker = self.epoch_rewards_workers[i, :episode_no]
            moving_avg_workers.append(moving_average(rewards_of_this_worker, window_size))
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_rewards, label="Episode Rewards", alpha=0.5)
        plt.plot(moving_avg_central, 
                label=f"{window_size}-Episode Moving Average",
                linewidth=2)
        for i in range(self.number_of_agents):
            plt.plot(moving_avg_workers[i], 
                    label=f"Worker {i} Rewards",
                    linewidth=0.5,
                    linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Rewards Over Training")
        plt.legend()
        plt.grid()
        plt.savefig(save_path_rewards, format='png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_rewards, label="Episode Rewards", alpha=0.5)
        plt.plot(moving_avg_central, 
                label=f"{window_size}-Episode Moving Average",
                linewidth=2)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Central Agent Rewards Over Training")
        plt.legend()
        plt.grid()
        plt.savefig(save_path_rewards_central, format='png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        
        for i in range(self.number_of_agents):
            plt.subplot(1, self.number_of_agents, i+1)
            plt.plot(moving_avg_workers[i], 
                    label=f"Worker {i} Rewards",
                    linewidth=0.5,
                    linestyle='--')
            plt.xlabel("Episodes")
            plt.ylabel("Total Reward")
            plt.title(f"Worker {i}")
            plt.grid()
        plt.tight_layout()
        plt.savefig(save_path_rewards_workers, format='png', dpi=300)
        plt.close()

    def load_all(self, central_agent, worker_agents):
        self.central_agent = central_agent
        self.worker_agents = worker_agents

    def save_all(self, episode_no):
        # Plot rewards
        self.plot_rewards(episode_no)
        # Plot agent logs
        self.central_agent.plotter()
        # Save agent and buffer
        self.central_agent.save(self.info)
        for i, worker_agent in enumerate(self.worker_agents):
            worker_agent.save(self.info + f'_worker_{i}')
        # If a test_env function is defined, run it
        if hasattr(self, 'test_env'):
            self.test_env()

    def fill_replay_buffer(self):
        """
        Fill the replay buffer with initial experience using random actions.
        """
        for i, agent in enumerate(self.worker_agents):
            while len(agent.buffer) < agent.buffer.buffer_size:
                state = self.env.reset()
                done = False
                while not done:
                    action = self.env.action_space.sample()
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    self.worker_agents[i].buffer.add(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done or truncated,
                        td_error=0.0
                    )
                    state = next_state
        # Sample from each buffer to fill the central agent buffer
        # Equally from each agent
        agent_id = 0
        while len(self.central_agent.buffer) < self.central_agent.buffer.buffer_size:
            states, actions, rewards, next_states, dones, _, weights = self.worker_agents[agent_id].buffer(self.central_agent.rng_key)
            self.central_agent.buffer.add(
                state=states,
                action=actions,
                reward=rewards,
                next_state=next_states,
                done=dones,
                td_error=jnp.zeros_like(weights)
            )

            agent_id += 1
            if agent_id >= self.number_of_agents:
                agent_id = 0

    def calculate_td_error(self,
                           selected_agent,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        return selected_agent.calculate_td_error(states, actions, rewards, next_states, dones)

    def individual_agent_update(self, selected_agent, state):
        action = selected_agent.select_actions(jnp.expand_dims(state, 0))
        next_state, reward, done, truncated, _ = self.env.step(action)
        done_or_truncated = done or truncated

        state_jnp = jnp.array(state)
        action_jnp = jnp.array(action)
        reward_jnp = jnp.array(reward)
        next_state_jnp = jnp.array(next_state)
        done_jnp = jnp.array(done_or_truncated)
        
        if action_jnp.ndim == 0:
            action_jnp = jnp.expand_dims(action_jnp, 0)

        td_error = self.calculate_td_error(
            selected_agent,
            jnp.expand_dims(state_jnp, axis=0),
            jnp.expand_dims(action_jnp, axis=0),
            jnp.expand_dims(reward_jnp, axis=0),
            jnp.expand_dims(next_state_jnp, axis=0),
            jnp.expand_dims(done_jnp, axis=0)
        )

        selected_agent.buffer.add(
            state=state_jnp,
            action=action_jnp,
            reward=reward_jnp,
            next_state=next_state_jnp,
            done=done_jnp,
            td_error= jnp.squeeze(td_error)
        )

        state = next_state

        selected_agent.update()
        
        return (selected_agent, next_state, \
                jnp.expand_dims(state_jnp, axis=0), \
                jnp.expand_dims(action_jnp, axis=0), \
                jnp.expand_dims(reward_jnp, axis=0), \
                jnp.expand_dims(next_state_jnp, axis=0), \
                jnp.expand_dims(done_jnp, axis=0), \
                jnp.squeeze(td_error))

    def add_buffer_and_update_worker_agents(self):
        worker_rewards = np.zeros(self.number_of_agents)
        for i, worker_agent in enumerate(self.worker_agents):
            state = self.env.reset()
            done_or_truncated = False
            reward_episode = 0
            while not (done_or_truncated):
                self.worker_agents[i], state, state_jnp, action_jnp, reward_jnp, next_state_jnp, done_jnp, td_error = self.individual_agent_update(self.worker_agents[i], state)
                # Add to central agent buffer
                self.central_agent.buffer.add(
                    state=state_jnp,
                    action=action_jnp,
                    reward=reward_jnp,
                    next_state=next_state_jnp,
                    done=done_jnp,
                    td_error= td_error
                )
                done_or_truncated = done_jnp
                reward_episode += reward_jnp[0]
            worker_rewards[i] += reward_episode
        return worker_rewards
    
    def train(self):
        self.fill_replay_buffer()

        while len(self.central_agent.buffer) < self.central_agent.buffer.buffer_size:
            _ = self.add_buffer_and_update_worker_agents()

        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")
        
        for episode in pbar:
            state = self.env.reset()
            done_or_truncated = False
            total_reward = 0
            num_steps = 0

            while not (done_or_truncated):
                self.central_agent, state, _, action, reward, _, done_or_truncated, _ = self.individual_agent_update(self.central_agent, state)

                total_reward += reward
                num_steps += 1
                if done_or_truncated:
                    self.central_agent.update_episode()

            worker_rewards = self.add_buffer_and_update_worker_agents()
            for i, worker_agent in enumerate(self.worker_agents):
                self.epoch_rewards_workers[i, episode] += worker_rewards[i]
            # Log the total reward for the episode
            self.epoch_rewards.append(total_reward)
            pbar.set_description(f"Training Progress - Episode: {episode}, Total Reward: {total_reward:.2f}, Num Steps: {num_steps}:")
            # Plot the rewards and losses
            if episode % self.save_interval == 0:
                self.save_all(episode)

        self.save_all()
        print("Training complete.")

#### MARL - CTDE ####

class Trainer_MARL_CTDE:
    def __init__(self,
                 env,
                 marl_ctde_agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 info: str = ""):
        
        self.env = env
        self.marl_ctde_agent = marl_ctde_agent
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.info = info
        
        # Initialize rewards list for each agent
        self.epoch_rewards = [[] for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
        # Initialize steps list for each agent
        self.steps_per_episode = [[] for _ in range(self.marl_ctde_agent.number_of_workers + 1)]

    def save_all(self):
        self.plot_rewards()
        self.marl_ctde_agent.plotter()
        self.marl_ctde_agent.save(self.info)
        if hasattr(self, 'test_env'):
            self.test_env()

    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        return self.marl_ctde_agent.calculate_td_error(states, actions, rewards, next_states, dones)

    def reset(self):
        self.epoch_rewards = [[] for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
        self.steps_per_episode = [[] for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
        self.marl_ctde_agent.reset()
        self.env.reset()

    def fill_replay_buffer(self):
        while len(self.marl_ctde_agent.buffer) < self.marl_ctde_agent.buffer.buffer_size:
            state = self.env.reset()
            done = False
            while not done:
                # Replace constant action with random sampling
                action = self.env.action_space.sample()
                action = jnp.array(action)  # Convert to jax array
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.marl_ctde_agent.buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated,
                    td_error=0.0
                )
                state = next_state        

    def train(self):
        self.fill_replay_buffer()
        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")

        # Create separate environments for each worker and the central agent
        envs = []
        for _ in range(self.marl_ctde_agent.number_of_workers + 1):
            # New instance of self.env
            env_new = type(self.env)(sizing_needed_bool = False) # This is exclusively for VR :()
            envs.append(env_new)

        for episode in pbar:
            states = [env.reset() for env in envs]
            done = [False for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
            truncated = [False for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
            total_reward = [0 for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
            num_steps = [0 for _ in range(self.marl_ctde_agent.number_of_workers + 1)]
            done_or_truncated = [False for _ in range(self.marl_ctde_agent.number_of_workers + 1)]

            while not all(done_or_truncated):
                actions = self.marl_ctde_agent.select_actions_from_all_workers(states)

                # Step workers
                for i in range(self.marl_ctde_agent.number_of_workers):
                    next_state, reward, done[i], truncated[i], _ = envs[i].step(actions[i])
                    done_or_truncated[i] = done[i] or truncated[i]
                    total_reward[i] += reward
                    num_steps[i] += 1

                    state_jnp = jnp.array(states[i])
                    action_jnp = jnp.array(actions[i])
                    reward_jnp = jnp.array(reward)
                    next_state_jnp = jnp.array(next_state)
                    done_jnp = jnp.array(done_or_truncated[i])

                    if action_jnp.ndim == 0:
                        action_jnp = jnp.expand_dims(action_jnp, axis=0)

                    td_error = self.calculate_td_error(
                        jnp.expand_dims(state_jnp, axis=0),
                        jnp.expand_dims(action_jnp, axis=0),
                        jnp.expand_dims(reward_jnp, axis=0),
                        jnp.expand_dims(next_state_jnp, axis=0),
                        jnp.expand_dims(done_jnp, axis=0)
                    )

                    self.marl_ctde_agent.buffer.add(
                        state=state_jnp,
                        action=action_jnp,
                        reward=reward_jnp,
                        next_state=next_state_jnp,
                        done=done_jnp,
                        td_error=jnp.squeeze(td_error)
                    )

                    states[i] = next_state

                # Step central agent
                next_state, reward, done[-1], truncated[-1], _ = envs[-1].step(actions[-1])
                done_or_truncated[-1] = done[-1] or truncated[-1]
                total_reward[-1] += reward
                num_steps[-1] += 1
                states[-1] = next_state

                # Update central agent
                self.marl_ctde_agent.update()

            # Only call update_episode at the end of each episode
            self.marl_ctde_agent.update_episode()

            # Store rewards and steps for each agent separately
            for i in range(len(total_reward)):
                self.epoch_rewards[i].append(total_reward[i])
                self.steps_per_episode[i].append(num_steps[i])

            pbar.set_description(f"Training Progress - Episode: {episode}, Total Reward: {total_reward:.2f}, Num Steps: {num_steps}:")
            if episode % self.save_interval == 0:
                self.save_all()

        self.save_all()
        print("Training complete.")
            
                
    def plot_rewards(self):
        '''
        self.epoch_rewards :
        [[episode_1_worker_1_reward, episode_1_worker_2_reward, ...],
        [episode_2_worker_1_reward, episode_2_worker_2_reward, ...],
        ...]
        so
        epoch_rewards_agent :
        [episode_1_worker_1_reward, episode_1_worker_2_reward, ...]
        '''
        save_path_rewards = self.marl_ctde_agent.save_path + 'rewards.png'
        plt.figure()
        for agent_id in range(self.marl_ctde_agent.number_of_workers+1): # So workers + central agent
            epoch_rewards_agent = self.epoch_rewards[agent_id]
            moving_av = moving_average(epoch_rewards_agent)
            if agent_id == self.marl_ctde_agent.number_of_workers:
                plt.plot(moving_av, label=f'Central Agent')
            else:
                plt.plot(moving_av, label=f'Worker {agent_id}')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards over Time')
        plt.grid()
        plt.savefig(save_path_rewards)
        plt.close()

        # Plot rewards per step using steps_per_episode
        plt.figure()
        for agent_id in range(self.marl_ctde_agent.number_of_workers+1):
            rewards_per_step = np.array(self.epoch_rewards[agent_id]) / np.array(self.steps_per_episode[agent_id])
            plt.plot(rewards_per_step, label=f'Agent {agent_id}')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Reward per Step')
        plt.title('Rewards per Step over Time')
        plt.grid()
        plt.savefig(self.marl_ctde_agent.save_path + 'rewards_per_step.png')
        plt.close()

    def test_env(self):
        pass