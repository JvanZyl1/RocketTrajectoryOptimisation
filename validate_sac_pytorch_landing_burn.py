import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import json

from src.agents.sac_pytorch import SACPyTorch
from src.envs.rl.env_wrapped_rl_pytorch import rl_wrapped_env_pytorch
from src.envs.rl.env_wrapped_rl_pytorch import maximum_velocity as maximum_velocity_lambda
from src.envs.universal_physics_plotter import universal_physics_plotter

# Monkey patch to add required methods for compatibility with universal_physics_plotter
class SACPyTorchWithPlotterSupport(SACPyTorch):
    def select_actions_no_stochastic(self, state):
        """Deterministic action selection for compatibility with universal_physics_plotter"""
        return self.select_action(state, deterministic=True)
    
    def select_actions(self, state):
        """Stochastic action selection for compatibility with universal_physics_plotter"""
        return self.select_action(state)

def main(tau_val = 0.005):
    # Create unique run ID for this training session
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Create a single run directory for all files related to this run
    run_dir = f"data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/{run_id}"
    os.makedirs(f"{run_dir}/agent_saves", exist_ok=True)
    os.makedirs(f"{run_dir}/learning_stats", exist_ok=True)
    os.makedirs(f"{run_dir}/trajectories", exist_ok=True)
    os.makedirs(f"{run_dir}/metrics", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)
    
    # Also create the traditional directories for backward compatibility
    os.makedirs("results/PyTorchSAC/LandingBurnPureThrottle", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/saves", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/runs", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/learning_stats", exist_ok=True)
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 2000
    evaluation_frequency = 10
    num_eval_episodes = 5
    save_stats_frequency = 50  # Frequency to save learning statistics
    
    # For quick verification of logging at start of training
    early_log_check = True  # Set to True to check logging early in training
    early_log_check_frequency = 10  # Save stats every 10 updates for early verification
    
    # Hyperparameters for SAC
    sac_hyperparams = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim_actor": 256,
        "number_of_hidden_layers_actor": 2,
        "hidden_dim_critic": 256,
        "number_of_hidden_layers_critic": 2,
        "alpha_initial": 0.2,
        "gamma": 0.99,
        "tau": tau_val,
        "buffer_size": 100000,
        "batch_size": 256,
        "critic_learning_rate": 0.005,
        "actor_learning_rate": 0.0001,
        "alpha_learning_rate": 3e-4,
        "max_action": 1.0,
        "auto_entropy_tuning": True
    }
    
    # Initialize the PyTorch SAC agent with hyperparameters
    agent = SACPyTorchWithPlotterSupport(
        state_dim=sac_hyperparams["state_dim"],
        action_dim=sac_hyperparams["action_dim"],
        # Network architecture
        hidden_dim_actor=sac_hyperparams["hidden_dim_actor"],
        number_of_hidden_layers_actor=sac_hyperparams["number_of_hidden_layers_actor"],
        hidden_dim_critic=sac_hyperparams["hidden_dim_critic"],
        number_of_hidden_layers_critic=sac_hyperparams["number_of_hidden_layers_critic"],
        # Hyperparameters
        alpha_initial=sac_hyperparams["alpha_initial"],
        gamma=sac_hyperparams["gamma"],
        tau=sac_hyperparams["tau"],
        buffer_size=sac_hyperparams["buffer_size"],
        batch_size=sac_hyperparams["batch_size"],
        # Learning rates
        critic_learning_rate=sac_hyperparams["critic_learning_rate"],
        actor_learning_rate=sac_hyperparams["actor_learning_rate"],
        alpha_learning_rate=sac_hyperparams["alpha_learning_rate"],
        # Action limit (should be 1.0 for normalized actions)
        max_action=sac_hyperparams["max_action"],
        # Flight phase (used for saving)
        flight_phase="LandingBurnPureThrottle",
        # Enable automatic entropy tuning
        auto_entropy_tuning=sac_hyperparams["auto_entropy_tuning"],
        # Learning stats save frequency
        save_stats_frequency=save_stats_frequency
    )
    
    # Set the run_id as an attribute of the agent so it can be used in update method
    agent.run_id = run_id
    
    # Add training configuration to hyperparameters
    training_config = {
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "evaluation_frequency": evaluation_frequency,
        "num_eval_episodes": num_eval_episodes,
        "save_stats_frequency": save_stats_frequency,
        "environment": {
            "flight_phase": "landing_burn_pure_throttle",
            "enable_wind": False,
            "stochastic_wind": False,
            "trajectory_length": 1000,
            "discount_factor": 0.99
        },
        "run_id": run_id
    }
    
    # Combine hyperparameters and training config
    all_params = {
        "sac_hyperparameters": sac_hyperparams,
        "training_config": training_config
    }
    
    # Save hyperparameters to JSON file
    hyperparams_path = f"{run_dir}/hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(all_params, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Create CSV files with headers for training and evaluation metrics
    training_metrics_path = f"{run_dir}/metrics/training_metrics.csv"
    eval_metrics_path = f"{run_dir}/metrics/eval_metrics.csv"
    
    # Create training metrics CSV with header
    pd.DataFrame(columns=['episode', 'reward', 'steps', 'updates']).to_csv(training_metrics_path, index=False)
    
    # Create evaluation metrics CSV with header
    pd.DataFrame(columns=['episode', 'eval_reward', 'success_rate']).to_csv(eval_metrics_path, index=False)
    
    # Create a learning stats file with header that will be appended to
    learning_stats_path = f"{run_dir}/learning_stats/sac_learning_stats.csv"
    if not os.path.exists(learning_stats_path):
        # Create an empty DataFrame with the same columns as learning_stats
        # This ensures headers are written before the agent starts appending
        empty_stats = {key: [] for key in agent.learning_stats.keys()}
        pd.DataFrame(empty_stats).to_csv(learning_stats_path, index=False)
        print(f"Created learning stats file: {learning_stats_path}")
    
    # If checking logs early, temporarily set to a smaller value
    if early_log_check:
        agent.save_stats_frequency = early_log_check_frequency
        print(f"Early log check enabled: Saving learning stats every {early_log_check_frequency} updates")
    
    # Override the agent's save paths to use our run directory
    agent.run_dir = run_dir
    
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
        #print(f"Episode {episode}: Reward = {float(episode_reward):.2f}, Steps = {step+1}, Updates = {updates_this_episode}")
        
        # Append this episode's data to the training metrics CSV
        episode_data = pd.DataFrame({
            'episode': [episode + 1],
            'reward': [float(episode_reward)],
            'steps': [step + 1],
            'updates': [updates_this_episode]
        })
        episode_data.to_csv(training_metrics_path, mode='a', header=False, index=False)
        
        # Early logging check: Save and restore normal frequency after a few episodes
        if early_log_check and episode == 2:
            print("Performing early logging check...")
            # Force save learning stats
            agent.save_learning_stats(run_id=run_id)
            # Restore normal frequency
            agent.save_stats_frequency = save_stats_frequency
            print(f"Early logging check complete. Restored normal save frequency to {save_stats_frequency}")
            early_log_check = False
        
        # Evaluation phase
        if (episode + 1) % evaluation_frequency == 0:
            eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes)
            eval_rewards.append(eval_reward)
            landing_success_rate.append(success_rate)
            print(f"Evaluation after episode {episode}: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
            
            # Append evaluation metrics to CSV
            eval_data = pd.DataFrame({
                'episode': [episode + 1],
                'eval_reward': [eval_reward],
                'success_rate': [success_rate]
            })
            eval_data.to_csv(eval_metrics_path, mode='a', header=False, index=False)
            
            # Save agent periodically
            if (episode + 1) % (evaluation_frequency * 5) == 0:
                # Save the agent, which will also save learning statistics
                agent.save(run_id=run_id)
                
                # Visualize a trajectory with the current agent
                visualize_trajectory(agent, env, run_id=run_id, run_dir=run_dir)
                plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=run_dir)
    
    # Final evaluation
    eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes * 2)
    print(f"Final evaluation: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
    
    # Append final evaluation metrics to CSV
    final_eval_data = pd.DataFrame({
        'episode': [num_episodes],
        'eval_reward': [eval_reward],
        'success_rate': [success_rate]
    })
    final_eval_data.to_csv(eval_metrics_path, mode='a', header=False, index=False)
    
    # Save the trained agent
    agent.save(run_id=run_id)
    
    # Plot training results
    plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=run_dir)
    
    # Visualize a final trajectory
    visualize_trajectory(agent, env, run_id=run_id, run_dir=run_dir, final=True)
    
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

def save_trajectory_to_csv(states, actions, rewards, infos, dynamic_pressure, throttle_command, maximum_velocity, run_id, run_dir=None, final=False):
    """
    Save trajectory data to a CSV file.
    
    Args:
        states: Array of state variables
        actions: Array of actions
        rewards: Array of rewards
        infos: List of info dictionaries
        dynamic_pressure: Array of dynamic pressure values
        throttle_command: Array of throttle commands
        maximum_velocity: Array of maximum velocity values
        run_id: Unique run identifier
        run_dir: Directory for this run's data
        final: Whether this is the final trajectory of training
    
    Returns:
        csv_path: Path to the saved CSV file
    """
    if run_dir:
        if final:
            csv_path = f"{run_dir}/trajectories/trajectory_final.csv"
        else:
            csv_path = f"{run_dir}/trajectories/trajectory.csv"
    else:
        # Legacy path
        if final:
            csv_path = f"data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/trajectory_final_{run_id}.csv"
        else:
            csv_path = f"data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/trajectory_{run_id}.csv"
    
    # Get the number of states and steps
    num_states = len(states)
    num_steps = len(rewards)  # rewards is one less than states
    
    # Create DataFrame with properly labeled state variables
    # Based on state definition: x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
    data = {
        'time': states[:, 10],
        'x': states[:, 0],            # horizontal position
        'y': states[:, 1],            # altitude
        'vx': states[:, 2],           # horizontal velocity
        'vy': states[:, 3],           # vertical velocity
        'theta': states[:, 4],        # pitch angle
        'theta_dot': states[:, 5],    # pitch rate
        'gamma': states[:, 6],        # flight path angle
        'alpha': states[:, 7],        # angle of attack
        'mass': states[:, 8],         # total mass
        'mass_propellant': states[:, 9],  # propellant mass
    }
    
    # Add the action and reward data for the steps (one less than states)
    # For the final state, we'll repeat the last values
    if num_steps > 0:  # Make sure we have actions/rewards
        # Use zeros for primary trajectory data as it's expected to have valid values
        data['throttle'] = np.zeros(num_states)
        data['throttle'][:num_steps] = throttle_command
        data['throttle'][num_steps:] = throttle_command[-1] if num_steps > 0 else 0
        
        data['action'] = np.zeros(num_states)
        flattened_actions = np.array([a.flatten()[0] for a in actions])
        data['action'][:num_steps] = flattened_actions
        data['action'][num_steps:] = flattened_actions[-1] if num_steps > 0 else 0
        
        data['reward'] = np.zeros(num_states)
        data['reward'][:num_steps] = rewards
        # Last state has no reward
    
    # Add other metrics, ensuring all arrays are the same length
    data['dynamic_pressure'] = np.zeros(num_states)
    data['dynamic_pressure'][:len(dynamic_pressure)] = dynamic_pressure
    if len(dynamic_pressure) < num_states and len(dynamic_pressure) > 0:
        data['dynamic_pressure'][len(dynamic_pressure):] = dynamic_pressure[-1]
        
    data['maximum_velocity'] = np.zeros(num_states)
    data['maximum_velocity'][:len(maximum_velocity)] = maximum_velocity
    if len(maximum_velocity) < num_states and len(maximum_velocity) > 0:
        data['maximum_velocity'][len(maximum_velocity):] = maximum_velocity[-1]
    
    # Automatically extract all keys and nested keys from the info dictionary
    if len(infos) > 0:
        # First, collect all available keys in all info dictionaries
        all_scalar_keys = set()
        all_nested_keys = {}  # {dict_key: set(scalar_subkeys)}
        
        # Find all available keys first
        for info in infos:
            for key, value in info.items():
                if key in ['state', 'actions']:
                    continue
                
                if isinstance(value, dict):
                    if key not in all_nested_keys:
                        all_nested_keys[key] = set()
                    
                    for subkey, subvalue in value.items():
                        if subkey == 'throttle':  # Skip throttle as we already have it
                            continue
                        
                        if np.isscalar(subvalue):
                            all_nested_keys[key].add(subkey)
                elif np.isscalar(value):
                    all_scalar_keys.add(key)
        
        # Now initialize arrays for all keys with proper length
        # Use NaN for info dictionary fields since these might be missing in some steps
        for key in all_scalar_keys:
            if key not in data:
                data[key] = np.full(num_states, np.nan)
        
        for dict_key, subkeys in all_nested_keys.items():
            for subkey in subkeys:
                col_name = f"{dict_key}_{subkey}"
                if col_name not in data:
                    data[col_name] = np.full(num_states, np.nan)
        
        # Now fill in the data
        for info_idx, info in enumerate(infos):
            if info_idx >= num_states:
                break  # Safety check
                
            # Fill in scalar values
            for key in all_scalar_keys:
                if key in info and np.isscalar(info[key]):
                    data[key][info_idx] = info[key]
            
            # Fill in nested dictionary values
            for dict_key, subkeys in all_nested_keys.items():
                if dict_key in info and isinstance(info[dict_key], dict):
                    for subkey in subkeys:
                        col_name = f"{dict_key}_{subkey}"
                        if subkey in info[dict_key] and np.isscalar(info[dict_key][subkey]):
                            data[col_name][info_idx] = info[dict_key][subkey]
    
    # Safety check - verify all arrays have the same length
    lengths = [len(arr) for arr in data.values()]
    if len(set(lengths)) > 1:
        print(f"Warning: Arrays have different lengths: {dict(zip(data.keys(), lengths))}")
        # Adjust all arrays to the minimum length
        min_length = min(lengths)
        data = {k: v[:min_length] for k, v in data.items()}
    
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Save to CSV - overwrite existing file if it exists
    df.to_csv(csv_path, index=False)
    print(f"Trajectory data saved to {csv_path}")
    
    return csv_path

def visualize_trajectory(agent, env, run_id, run_dir=None, save_path=None, final=False):
    """Visualize a trajectory using the agent"""
    # Create a simple visualization
    state = env.reset()
    states = []
    actions = []
    rewards = []
    dynamic_pressure = []
    throttle_command = []
    maximum_velocity = []
    infos = []  # Store all info dictionaries
    
    done = False
    truncated = False
    
    while not (done or truncated):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, truncated, info = env.step(action)
        
        raw_state = info['state']
        states.append(raw_state)
        actions.append(action)
        rewards.append(reward)
        dynamic_pressure.append(info['dynamic_pressure'])
        throttle_command.append(info['action_info']['throttle'])
        maximum_velocity_state = maximum_velocity_lambda(raw_state[1], raw_state[3])
        maximum_velocity.append(maximum_velocity_state)
        infos.append(info)  # Save the entire info dictionary
        state = next_state
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dynamic_pressure = np.array(dynamic_pressure)
    throttle_command = np.array(throttle_command)
    maximum_velocity = np.array(maximum_velocity)
    
    # Save trajectory data to CSV
    csv_path = save_trajectory_to_csv(
        states=states,
        actions=actions,
        rewards=rewards,
        infos=infos,
        dynamic_pressure=dynamic_pressure,
        throttle_command=throttle_command,
        maximum_velocity=maximum_velocity,
        run_id=run_id,
        run_dir=run_dir,
        final=final
    )
    
    # Plot trajectory
    plt.figure(figsize=(15, 10))
    plt.suptitle('Landing Burn Trajectory', fontsize=16)
    
    # Plot altitude vs time
    plt.subplot(3, 2, 1)
    plt.plot(states[:, 10], states[:, 1], color='blue', linewidth=2)  # time vs y
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.title('Altitude vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)

    # Plot lateral position vs time
    plt.subplot(3, 2, 2)
    plt.plot(states[:, 10], states[:, 0], color='blue', linewidth=2)  # time vs x
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Horizontal Position (m)', fontsize=16)
    plt.title('Horizontal Position vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Plot vertical velocity vs time
    plt.subplot(3, 2, 3)
    plt.plot(states[:, 10], states[:, 3], color='blue', linewidth=2)  # time vs vy
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Vertical Velocity (m/s)', fontsize=16)
    plt.title('Vertical Velocity vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)

    # Plot lateral velocity vs time
    plt.subplot(3, 2, 4)
    plt.plot(states[:, 10], states[:, 2], color='blue', linewidth=2)  # time vs vx
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Horizontal Velocity (m/s)', fontsize=16)
    plt.title('Horizontal Velocity vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Plot throttle command vs time
    plt.subplot(3, 2, 5)
    plt.plot(states[:-1, 10], throttle_command[:len(states)-1], color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Throttle Command', fontsize=16)
    plt.title('Throttle Command vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Plot rewards vs time
    plt.subplot(3, 2, 6)
    plt.plot(states[:-1, 10], rewards[:len(states)-1], color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.title('Reward vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        if run_dir:
            if final:
                plt.savefig(f"{run_dir}/plots/trajectory_final.png")
            else:
                plt.savefig(f"{run_dir}/plots/trajectory.png")
        else:
            # Legacy path
            if final:
                plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/trajectory_final_{run_id}.png")
            else:
                plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/trajectory_{run_id}.png")
    
    plt.close()

    # Plot time vs dynamic pressure
    plt.figure(figsize=(15, 10))
    plt.plot(states[:, 10], dynamic_pressure/1000, color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Dynamic Pressure (kPa)', fontsize=16)
    plt.title('Dynamic Pressure vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Save the dynamic pressure plot
    if save_path:
        plt.savefig(save_path)
    else:
        if run_dir:
            if final:
                plt.savefig(f"{run_dir}/plots/dynamic_pressure_final.png")
            else:
                plt.savefig(f"{run_dir}/plots/dynamic_pressure.png")
        else:
            # Legacy path
            if final:
                plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/dynamic_pressure_final_{run_id}.png")
            else:
                plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/dynamic_pressure_{run_id}.png")
    
    plt.close()
    
    # Create a separate X-Y trajectory plot
    plt.figure(figsize=(10, 8))
    plt.plot(states[:, 0], states[:, 1], color='blue', linewidth=2)  # x vs y
    plt.scatter(states[0, 0], states[0, 1], color='green', s=100, marker='o', label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], color='red', s=100, marker='x', label='End')
    plt.xlabel('Horizontal Position (m)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.title('X-Y Trajectory', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    
    # Save the X-Y trajectory plot
    if run_dir:
        if final:
            plt.savefig(f"{run_dir}/plots/xy_trajectory_final.png")
        else:
            plt.savefig(f"{run_dir}/plots/xy_trajectory.png")
    else:
        # Legacy path
        if final:
            plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/xy_trajectory_final_{run_id}.png")
        else:
            plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/xy_trajectory_{run_id}.png")
    
    plt.close()
    
    # Return total reward achieved
    total_reward = sum(rewards)
    print(f"Visualization completed. Total reward: {total_reward:.2f}")
    return total_reward, states[:, 1]  # Return reward and altitude array

def plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=None):
    """Plot training and evaluation metrics from saved CSV files"""
    # Load metrics from CSV files
    training_df = pd.read_csv(training_metrics_path)
    eval_df = pd.read_csv(eval_metrics_path)
    
    # Plot training curves
    plt.figure(figsize=(15, 12))
    
    # Plot training rewards
    plt.subplot(2, 2, 1)
    plt.plot(training_df['episode'], training_df['reward'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    plt.plot(eval_df['episode'], eval_df['eval_reward'], marker='o', linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.title('Evaluation Rewards', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    
    # Plot episode steps
    plt.subplot(2, 2, 3)
    plt.plot(training_df['episode'], training_df['steps'], linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Steps', fontsize=16)
    plt.title('Episode Length', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    
    # Plot landing success rate
    plt.subplot(2, 2, 4)
    plt.plot(eval_df['episode'], eval_df['success_rate'], marker='o', linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.title('Landing Success Rate', fontsize=16)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    
    if run_dir:
        plt.savefig(f"{run_dir}/plots/training_results.png")
    else:
        # Legacy path
        plt.savefig(f"results/PyTorchSAC/LandingBurnPureThrottle/training_results_{run_id}.png")
    plt.close()

if __name__ == "__main__":
    tau_vals = [0.005, 0.001, 0.01, 0.1]
    for tau_val in tau_vals:
        main(tau_val) 