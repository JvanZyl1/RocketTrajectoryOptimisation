import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
from collections import deque
import random
from tqdm import tqdm
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.agents.soft_actor_critic import SoftActorCritic
from src.agents.td3 import TD3
from src.envs.supervisory.agent_load_supervisory import load_supervisory_actor
from configs.agent_config import config_landing_burn
import matplotlib.pyplot as plt
# Hyperparameters
BUFFER_SIZE = 2000
BATCH_SIZE = 64
SEQ_LEN = 10
HIDDEN_DIM = 128
OBS_DIM = 2
ACT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = env(flight_phase = 'landing_burn_pure_throttle',
            enable_wind = False,
            trajectory_length = 100,
            discount_factor = 0.99)

actor, actor_params, hidden_dim, hidden_layers = load_supervisory_actor("landing_burn_pure_throttle", "td3")
config_landing_burn['td3']['hidden_dim_actor'] = hidden_dim
config_landing_burn['td3']['number_of_hidden_layers_actor'] = hidden_layers

agent = TD3(state_dim=env.state_dim,
                action_dim=env.action_dim,
                flight_phase = 'landing_burn_pure_throttle',
                **config_landing_burn['td3'])


# Replay Buffer with sequence support
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, td_error):
        self.buffer.append((state, action, reward, next_state, done, td_error))

    def sample_sequences(self, batch_size, seq_len):
        indices = np.random.randint(seq_len, len(self.buffer), size=batch_size)
        sequences = []
        for idx in indices:
            seq = list(self.buffer)[idx - seq_len:idx]
            obs_seq = torch.tensor([s[0] for s in seq], dtype=torch.float32)
            act = torch.tensor(seq[-1][1], dtype=torch.float32)
            rew = torch.tensor(seq[-1][2], dtype=torch.float32)
            next_obs = torch.tensor(seq[-1][3], dtype=torch.float32)
            done = torch.tensor(seq[-1][4], dtype=torch.float32)
            sequences.append((obs_seq, act, rew, next_obs, done))
        return sequences

# Fill buffer using JAX-like rollout logic
def fill_buffer(buffer, agent, env, buffer_size=BUFFER_SIZE, batch_size=100):
    experiences_batch = []
    non_zero_experiences = 0
    pbar = tqdm(total=buffer_size, desc="Filling replay buffer")

    while non_zero_experiences < buffer_size:
        state = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_actions(jnp.expand_dims(state, 0))
            action = np.array([action])
            next_state, reward, done, truncated, _ = env.step(action)
            done_or_truncated = done or truncated

            # Extract innermost list of action ndim changes
            if action.ndim == 2:
                action = action[0]
            else:
                action = action

            experiences_batch.append((state, action, reward, next_state, done_or_truncated))
            non_zero_experiences += 1
            pbar.update(1)
            state = next_state

            if len(experiences_batch) >= batch_size or non_zero_experiences >= buffer_size:
                states = np.array([exp[0] for exp in experiences_batch])
                actions = np.array([exp[1] for exp in experiences_batch])
                rewards = np.array([exp[2] for exp in experiences_batch])
                next_states = np.array([exp[3] for exp in experiences_batch])
                dones = np.array([exp[4] for exp in experiences_batch])

                td_errors = agent.calculate_td_error_vmap(states, actions, rewards, next_states, dones)

                for i in range(len(experiences_batch)):
                    buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i], td_errors[i])

                experiences_batch = []

                if non_zero_experiences >= buffer_size:
                    break

# Double Critic
class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        def make_critic():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.Q1 = make_critic()
        self.Q2 = make_critic()

    def forward(self, obs, act):
        # obs shape: [batch_size, obs_dim]
        # act shape: [batch_size, act_dim]
        x = torch.cat([obs, act], dim=1)  # [batch_size, obs_dim + act_dim]
        return self.Q1(x), self.Q2(x)

# GRU Critic
class GRUCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_seq, act):
        # obs_seq shape: [batch_size, seq_len, obs_dim]
        # act shape: [batch_size, act_dim]
        
        # Get the final hidden state from GRU
        _, h = self.gru(obs_seq)  # h shape: [1, batch_size, hidden_dim]
        h = h.squeeze(0)  # Remove the first dimension -> [batch_size, hidden_dim]
        
        # Concatenate with actions
        x = torch.cat([h, act], dim=1)  # [batch_size, hidden_dim + act_dim]
        
        # Pass through fully connected layers
        return self.fc(x)  # [batch_size, 1]

# Loss functions
def double_critic_loss(model, batch, target_model, gamma=0.99):
    # Extract and stack the final observation from each sequence
    obs_batch = torch.stack([obs_seq[-1] for obs_seq, _, _, _, _ in batch])  # [batch_size, obs_dim]
    act_batch = torch.stack([act for _, act, _, _, _ in batch])              # [batch_size, act_dim]
    rew_batch = torch.stack([rew for _, _, rew, _, _ in batch]).unsqueeze(1) # [batch_size, 1]
    next_obs_batch = torch.stack([next_obs for _, _, _, next_obs, _ in batch]) # [batch_size, obs_dim]
    done_batch = torch.stack([done for _, _, _, _, done in batch]).unsqueeze(1) # [batch_size, 1]
    
    # Get target Q values
    with torch.no_grad():
        target_q1, target_q2 = target_model(next_obs_batch, act_batch)
        target_q = rew_batch + gamma * (1 - done_batch) * torch.min(target_q1, target_q2)
    
    # Get current Q values
    q1, q2 = model(obs_batch, act_batch)
    
    # Compute loss
    loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    
    return loss

def gru_critic_loss(model, batch, target_model, gamma=0.99):
    # Create a batch of sequence data
    obs_seqs = torch.stack([obs_seq for obs_seq, _, _, _, _ in batch])  # [batch_size, seq_len, obs_dim]
    act_batch = torch.stack([act for _, act, _, _, _ in batch])         # [batch_size, act_dim]
    rew_batch = torch.stack([rew for _, _, rew, _, _ in batch]).unsqueeze(1)  # [batch_size, 1]
    next_obs_batch = torch.stack([next_obs for _, _, _, next_obs, _ in batch])  # [batch_size, obs_dim]
    done_batch = torch.stack([done for _, _, _, _, done in batch]).unsqueeze(1)  # [batch_size, 1]
    
    # Create next_input by combining the sequence except first element with next_obs
    next_seqs = torch.cat([
        obs_seqs[:, 1:, :],  # Remove first observation from sequence
        next_obs_batch.unsqueeze(1)  # Add next_obs as last element
    ], dim=1)  # [batch_size, seq_len, obs_dim]
    
    with torch.no_grad():
        target_q = rew_batch + gamma * (1 - done_batch) * target_model(next_seqs, act_batch)
    
    pred_q = model(obs_seqs, act_batch)
    loss = F.mse_loss(pred_q, target_q)
    
    return loss

# Main execution
def main():
    buffer = ExperienceBuffer(BUFFER_SIZE)
    fill_buffer(buffer, agent, env)

    # Create models
    double_critic = DoubleCritic(OBS_DIM, ACT_DIM).to(DEVICE)
    target_double_critic = DoubleCritic(OBS_DIM, ACT_DIM).to(DEVICE)
    target_double_critic.load_state_dict(double_critic.state_dict())
    
    gru_critic = GRUCritic(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(DEVICE)
    target_gru_critic = GRUCritic(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(DEVICE)
    target_gru_critic.load_state_dict(gru_critic.state_dict())
    
    # Optimizers
    dc_optimizer = torch.optim.Adam(double_critic.parameters(), lr=3e-3)
    gru_optimizer = torch.optim.Adam(gru_critic.parameters(), lr=3e-3)
    
    # Training loop
    num_steps = 10000
    dc_losses = []
    gru_losses = []
    
    for step in tqdm(range(num_steps), desc="Training critics"):
        batch = buffer.sample_sequences(BATCH_SIZE, SEQ_LEN)
        batch = [(obs.to(DEVICE), act.to(DEVICE), rew.to(DEVICE), next_obs.to(DEVICE), done.to(DEVICE)) 
                for obs, act, rew, next_obs, done in batch]
        
        # Train double critic
        dc_optimizer.zero_grad()
        dc_loss = double_critic_loss(double_critic, batch, target_double_critic)
        dc_loss.backward()
        dc_optimizer.step()
        dc_losses.append(dc_loss.item())
        
        # Train GRU critic
        gru_optimizer.zero_grad()
        gru_loss = gru_critic_loss(gru_critic, batch, target_gru_critic)
        gru_loss.backward()
        gru_optimizer.step()
        gru_losses.append(gru_loss.item())
        
        # Update target networks
        if step % 10 == 0:
            # Soft update of target networks
            tau = 0.005
            for param, target_param in zip(double_critic.parameters(), target_double_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for param, target_param in zip(gru_critic.parameters(), target_gru_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(dc_losses, label='Double Critic Loss')
    plt.plot(gru_losses, label='GRU Critic Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Critic Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('critic_loss_comparison.png')
    plt.show()
    
    print(f"Final Double Critic Loss: {dc_losses[-1]:.6f}")
    print(f"Final GRU Critic Loss:    {gru_losses[-1]:.6f}")

if __name__ == "__main__":
    main()
