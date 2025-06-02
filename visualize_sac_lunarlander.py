import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
import pickle
import os
import argparse

from src.agents.soft_actor_critic import SoftActorCritic

def load_agent(load_path):
    """Load a trained SAC agent from a pickle file"""
    with open(load_path, 'rb') as f:
        agent_state = pickle.load(f)
    
    # Create a new agent with the same parameters
    agent = SoftActorCritic(
        state_dim=agent_state['inputs']['state_dim'],
        action_dim=agent_state['inputs']['action_dim'],
        flight_phase=agent_state['inputs']['flight_phase'],
        hidden_dim_actor=agent_state['inputs']['hidden_dim_actor'],
        number_of_hidden_layers_actor=agent_state['inputs']['number_of_hidden_layers_actor'],
        hidden_dim_critic=agent_state['inputs']['hidden_dim_critic'],
        number_of_hidden_layers_critic=agent_state['inputs']['number_of_hidden_layers_critic'],
        temperature_initial=agent_state['inputs']['temperature_initial'],
        gamma=agent_state['inputs']['gamma'],
        tau=agent_state['inputs']['tau'],
        alpha_buffer=agent_state['inputs']['alpha_buffer'],
        beta_buffer=agent_state['inputs']['beta_buffer'],
        beta_decay_buffer=agent_state['inputs']['beta_decay_buffer'],
        buffer_size=agent_state['inputs']['buffer_size'],
        trajectory_length=agent_state['inputs']['trajectory_length'],
        batch_size=agent_state['inputs']['batch_size'],
        critic_learning_rate=agent_state['inputs']['critic_learning_rate'],
        actor_learning_rate=agent_state['inputs']['actor_learning_rate'],
        temperature_learning_rate=agent_state['inputs']['temperature_learning_rate'],
        critic_grad_max_norm=agent_state['inputs']['critic_grad_max_norm'],
        actor_grad_max_norm=agent_state['inputs']['actor_grad_max_norm'],
        temperature_grad_max_norm=agent_state['inputs']['temperature_grad_max_norm'],
        max_std=agent_state['inputs']['max_std'],
        l2_reg_coef=agent_state['inputs']['l2_reg_coef'],
        expected_updates_to_convergence=agent_state['inputs']['expected_updates_to_convergence']
    )
    
    # Restore the agent's parameters
    agent.critic_params = agent_state['update']['critic_params']
    agent.critic_target_params = agent_state['update']['critic_target_params']
    agent.actor_params = agent_state['update']['actor_params']
    agent.temperature = agent_state['update']['temperature']
    
    return agent

def visualize_agent(agent=None, load_path=None, num_episodes=5, seed=None):
    """Visualize the agent's performance in the environment"""
    if agent is None and load_path is not None:
        print(f"Loading agent from {load_path}")
        agent = load_agent(load_path)
    elif agent is None and load_path is None:
        raise ValueError("Either agent or load_path must be provided")
    
    # Create the environment with rendering
    if seed is not None:
        env = gym.make("LunarLander-v3", continuous=True, render_mode="human", seed=seed)
    else:
        env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = jnp.array(state, dtype=jnp.float32)
        episode_reward = 0
        done = False
        truncated = False
        step = 0
        
        print(f"Episode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            # Use deterministic action selection for visualization
            action = agent.select_actions_no_stochastic(state)
            action_np = np.array(action)
            
            next_state, reward, done, truncated, _ = env.step(action_np)
            state = jnp.array(next_state, dtype=jnp.float32)
            episode_reward += reward
            step += 1
            
            # Add a small delay to make the visualization easier to follow
            time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} finished with reward {episode_reward:.2f} in {step} steps")
    
    env.close()
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained SAC agent on LunarLander")
    parser.add_argument("--load_path", type=str, 
                        default="data/agent_saves/VanillaSAC/LunarLander/saves/soft-actor-critic.pkl",
                        help="Path to the saved agent")
    parser.add_argument("--episodes", type=int, default=5, 
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the environment")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.load_path):
        print(f"Warning: Agent file not found at {args.load_path}")
        print("You need to train an agent first using validate_sac_lunarlander.py")
        exit(1)
    
    visualize_agent(load_path=args.load_path, num_episodes=args.episodes, seed=args.seed) 