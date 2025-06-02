import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from datetime import datetime

from src.agents.sac_pytorch import SACPyTorch
from src.envs.rl.env_wrapped_rl_pytorch import rl_wrapped_env_pytorch
from src.envs.universal_physics_plotter import universal_physics_plotter

# Monkey patch to add required methods for compatibility with universal_physics_plotter
class SACPyTorchWithPlotterSupport(SACPyTorch):
    def select_actions_no_stochastic(self, state):
        """Deterministic action selection for compatibility with universal_physics_plotter"""
        return self.select_action(state, deterministic=True)
    
    def select_actions(self, state):
        """Stochastic action selection for compatibility with universal_physics_plotter"""
        return self.select_action(state)

def main():
    # Create the landing burn environment using PyTorch compatible wrapper
    env = rl_wrapped_env_pytorch(
        flight_phase="landing_burn_pure_throttle",
        enable_wind=False,
        stochastic_wind=False,
        trajectory_length=1000,
        discount_factor=0.99
    )
    
    # Environment info
    state_dim = env.state_dim  # Should be 2 for landing_burn_pure_throttle
    action_dim = env.action_dim  # Should be 1 for landing_burn_pure_throttle
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results/PyTorchSAC/LandingBurnPureThrottle", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/saves", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/runs", exist_ok=True)
    
    # Initialize the PyTorch SAC agent with appropriate hyperparameters
    agent = SACPyTorchWithPlotterSupport(
        state_dim=state_dim,
        action_dim=action_dim,
        # Network architecture
        hidden_dim_actor=256,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=256,
        number_of_hidden_layers_critic=2,
        # Hyperparameters
        alpha_initial=0.2,  # Initial temperature (entropy coefficient)
        gamma=0.99,         # Discount factor
        tau=0.005,          # Soft update coefficient
        buffer_size=100000, # Replay buffer size
        batch_size=256,     # Batch size for updates
        # Learning rates
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        # Action limit (should be 1.0 for normalized actions)
        max_action=1.0,
        # Flight phase (used for saving)
        flight_phase="LandingBurnPureThrottle",
        # Enable automatic entropy tuning
        auto_entropy_tuning=True
    )
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 1000
    evaluation_frequency = 10
    num_eval_episodes = 5
    
    # Lists to track progress
    episode_rewards = []
    eval_rewards = []
    episode_steps = []
    episode_updates = []
    landing_success_rate = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        updates_this_episode = 0
        
        for step in range(max_steps_per_episode):
            # Select action using the agent
            action = agent.select_action(state)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=1.0 if done else 0.0
            )
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Update the agent
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
                updates_this_episode += 1
            
            # End episode if done or truncated
            if done or truncated:
                break
        
        # Log episode results
        episode_rewards.append(float(episode_reward))
        episode_steps.append(step + 1)
        episode_updates.append(updates_this_episode)
        
        # Print episode stats
        print(f"Episode {episode}: Reward = {float(episode_reward):.2f}, Steps = {step+1}, Updates = {updates_this_episode}")
        
        # Evaluation phase
        if (episode + 1) % evaluation_frequency == 0:
            eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes)
            eval_rewards.append(eval_reward)
            landing_success_rate.append(success_rate)
            print(f"Evaluation after episode {episode}: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
            
            # Save agent periodically
            if (episode + 1) % (evaluation_frequency * 5) == 0:
                agent.save()
                
                # Visualize a trajectory with the current agent
                visualize_trajectory(agent, env)
    
    # Final evaluation
    eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes * 2)
    print(f"Final evaluation: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
    
    # Save the trained agent
    agent.save()
    
    # Plot training results
    plot_results(episode_rewards, eval_rewards, episode_steps, episode_updates, landing_success_rate, evaluation_frequency)
    
    # Visualize a final trajectory
    visualize_trajectory(agent, env)
    
    # Close environment
    env.close()

def evaluate_agent(agent, env, num_episodes):
    """Evaluate the agent without exploration"""
    eval_rewards = []
    successes = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Use deterministic action selection for evaluation
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            
            # Check if landing was successful (check y position and velocity)
            if done and not truncated:
                # Extract raw state (x, y, vx, vy, ...)
                raw_state = info['state']
                y = raw_state[1]  # altitude
                vy = raw_state[3]  # vertical velocity
                
                # Consider successful if altitude is near zero and velocity is low
                if y < 10 and abs(vy) < 3:
                    successes += 1
        
        eval_rewards.append(episode_reward)
    
    success_rate = successes / num_episodes
    return np.mean(eval_rewards), success_rate

def visualize_trajectory(agent, env, save_path=None):
    """Visualize a trajectory using the agent"""
    # Create a simple visualization
    state = env.reset()
    states = []
    actions = []
    rewards = []
    
    done = False
    truncated = False
    
    while not (done or truncated):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, truncated, info = env.step(action)
        
        raw_state = info['state']
        states.append(raw_state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Plot trajectory
    plt.figure(figsize=(15, 10))
    plt.suptitle('Landing Burn Trajectory', fontsize=16)
    
    # Plot altitude vs time
    plt.subplot(2, 2, 1)
    plt.plot(states[:, 10], states[:, 1], color='blue', linewidth=2)  # time vs y
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.title('Altitude vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Plot vertical velocity vs time
    plt.subplot(2, 2, 2)
    plt.plot(states[:, 10], states[:, 3])  # time vs vy
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Velocity (m/s)')
    plt.title('Vertical Velocity vs Time')
    plt.grid(True)
    
    # Plot throttle command vs time
    plt.subplot(2, 2, 3)
    plt.plot(states[:-1, 10], actions.flatten())
    plt.xlabel('Time (s)')
    plt.ylabel('Throttle Command')
    plt.title('Throttle Command vs Time')
    plt.grid(True)
    
    # Plot rewards vs time
    plt.subplot(2, 2, 4)
    plt.plot(states[:-1, 10], rewards)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title('Reward vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/trajectory_{timestamp}.png")
    
    plt.close()
    
    # Return total reward achieved
    total_reward = sum(rewards)
    print(f"Visualization completed. Total reward: {total_reward:.2f}")
    return total_reward, states[:, 1]  # Return reward and altitude array

def plot_results(episode_rewards, eval_rewards, episode_steps, episode_updates, landing_success_rate, eval_freq):
    """Plot training and evaluation metrics"""
    plt.figure(figsize=(15, 12))
    
    # Plot training rewards
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot evaluation rewards
    plt.subplot(3, 2, 2)
    eval_episodes = [(i+1) * eval_freq for i in range(len(eval_rewards))]
    plt.plot(eval_episodes, eval_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Rewards')
    plt.grid(True)
    
    # Plot episode steps
    plt.subplot(3, 2, 3)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length')
    plt.grid(True)
    
    # Plot updates per episode
    plt.subplot(3, 2, 4)
    plt.plot(episode_updates)
    plt.xlabel('Episode')
    plt.ylabel('Updates')
    plt.title('Updates per Episode')
    plt.grid(True)
    
    # Plot landing success rate
    plt.subplot(3, 2, 5)
    plt.plot(eval_episodes, landing_success_rate, marker='o', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Landing Success Rate')
    plt.ylim(0, 1.05)
    plt.grid(True)
    
    # Plot training curve with moving average
    plt.subplot(3, 2, 6)
    window_size = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Avg)')
    plt.title(f'Training Rewards ({window_size}-Episode Moving Average)')
    plt.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/training_results_{timestamp}.png")
    plt.show()

if __name__ == "__main__":
    main() 