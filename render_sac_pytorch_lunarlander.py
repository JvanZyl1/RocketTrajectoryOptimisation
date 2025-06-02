import gymnasium as gym
import numpy as np
import torch
import time
import argparse
import os

from src.agents.sac_pytorch import SACPyTorch

def main():
    parser = argparse.ArgumentParser(description='Render a trained SAC agent in LunarLander')
    parser.add_argument('--model_path', type=str, help='Path to the saved model', required=False)
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to render')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create the environment with rendering
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
    # Environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Max action value: {max_action}")
    print(f"Action range: [{env.action_space.low}, {env.action_space.high}]")
    
    # Initialize the agent
    agent = SACPyTorch(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        flight_phase="LunarLander"
    )
    
    # Load the trained model if path is provided
    if args.model_path:
        if os.path.exists(args.model_path):
            agent.load(args.model_path)
            print(f"Model loaded from {args.model_path}")
        else:
            print(f"Model file not found at {args.model_path}")
            print("Running with untrained agent")
    else:
        # Try to find the latest model in the default location
        save_dir = "data/agent_saves/PyTorchSAC/LunarLander/saves"
        if os.path.exists(save_dir):
            model_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("sac_pytorch_")]
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                agent.load(latest_model)
                print(f"Latest model loaded from {latest_model}")
            else:
                print("No model files found in default location. Running with untrained agent.")
        else:
            print("Default save directory not found. Running with untrained agent.")
    
    # Run episodes
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Select action deterministically
            action = agent.select_action(state, deterministic=True)
            
            # Take action in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Add a small delay for better visualization
            time.sleep(0.01)
        
        print(f"Episode {episode+1}/{args.episodes}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    env.close()

if __name__ == "__main__":
    main() 