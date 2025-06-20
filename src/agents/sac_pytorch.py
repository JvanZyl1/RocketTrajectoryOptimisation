import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
import os
from datetime import datetime
import pandas as pd
import scipy.stats as stats

class ReplayBuffer:
    """Simple replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0
        
        # Preallocate buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences from the buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )
    
    def __len__(self):
        return self.size

class Actor(nn.Module):
    """Actor network for SAC, outputs mean and log_std of the Gaussian policy."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        n_hidden_layers: int = 2,
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build the network
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.shared_net = nn.Sequential(*layers)
        
        # Mean and log_std outputs
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the actor network."""
        features = self.shared_net(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        
        # Constrain log_std to reasonable range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy given a state."""
        mean, log_std = self(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Calculate log probability of action
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action * self.max_action, log_prob

class Critic(nn.Module):
    """Critic network for SAC, outputs Q-value for a state-action pair."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        n_hidden_layers: int = 2
    ):
        super(Critic, self).__init__()
        
        # Build the Q1 network
        self.q1_layers = []
        self.q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        self.q1_layers.append(nn.ReLU())
        
        for _ in range(n_hidden_layers - 1):
            self.q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            
        self.q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*self.q1_layers)
        
        # Build the Q2 network (for mitigating overestimation bias)
        self.q2_layers = []
        self.q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        self.q2_layers.append(nn.ReLU())
        
        for _ in range(n_hidden_layers - 1):
            self.q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            
        self.q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*self.q2_layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of both critic networks."""
        state_action = torch.cat([state, action], dim=1)
        
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass of only the Q1 network."""
        state_action = torch.cat([state, action], dim=1)
        return self.q1(state_action)

class SACPyTorch:
    """Soft Actor-Critic implementation in PyTorch."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Actor network parameters
        hidden_dim_actor: int = 256,
        number_of_hidden_layers_actor: int = 2,
        # Critic network parameters
        hidden_dim_critic: int = 256,
        number_of_hidden_layers_critic: int = 2,
        # SAC hyperparameters
        alpha_initial: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        # Learning rates
        critic_learning_rate: float = 3e-4,
        actor_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # Action bounds
        max_action: float = 1.0,
        # Device to run on
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Flag for automatic entropy tuning
        auto_entropy_tuning: bool = True,
        # Flight phase
        flight_phase: str = "default",
        # Learning stats save frequency
        save_stats_frequency: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device
        self.auto_entropy_tuning = auto_entropy_tuning
        self.flight_phase = flight_phase
        self.save_stats_frequency = save_stats_frequency
        
        # Initialize networks
        self.actor = Actor(
            state_dim, 
            action_dim, 
            hidden_dim_actor, 
            number_of_hidden_layers_actor,
            max_action
        ).to(device)
        
        self.critic = Critic(
            state_dim, 
            action_dim, 
            hidden_dim_critic, 
            number_of_hidden_layers_critic
        ).to(device)
        
        self.critic_target = Critic(
            state_dim, 
            action_dim, 
            hidden_dim_critic, 
            number_of_hidden_layers_critic
        ).to(device)
        
        # Copy critic parameters to target critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        # Initialize temperature parameter alpha
        self.log_alpha = torch.tensor(np.log(alpha_initial), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        
        # If auto_entropy_tuning is enabled, set target entropy
        if auto_entropy_tuning:
            self.target_entropy = -action_dim  # heuristic -dim(A)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # Initialize update counter
        self.updates = 0
        
        # Create directories for saving
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/runs", exist_ok=True)
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/saves", exist_ok=True)
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/learning_stats", exist_ok=True)
        
        # Initialize learning statistics tracking
        self.learning_stats = {
            'step': [],
            # State statistics
            'state_mean': [],
            'state_min': [],
            'state_max': [],
            'state_std': [],
            'state_kurtosis': [],
            # Action statistics
            'action_mean': [],
            'action_min': [],
            'action_max': [],
            'action_std': [],
            'action_kurtosis': [],
            # Reward statistics
            'reward_mean': [],
            'reward_min': [],
            'reward_max': [],
            'reward_std': [],
            'reward_kurtosis': [],
            # Network statistics
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'q_value_mean': [],
            'q_value_min': [],
            'q_value_max': [],
            'q_value_std': [],
            'log_prob_mean': [],
            'log_prob_min': [],
            'log_prob_max': [],
            'log_prob_std': [],
            'target_q_mean': [],
            'target_q_min': [],
            'target_q_max': [],
            'target_q_std': [],
        }
    
    @property
    def alpha(self):
        """Get the current temperature parameter."""
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        """Select an action from the policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state, deterministic)
            return action.cpu().numpy().flatten()
    
    def update(self):
        """Update the networks using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter alpha
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update of the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        
        # Track learning statistics
        self._track_learning_stats(
            states=states,
            actions=actions,
            rewards=rewards,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            q_values=q,
            target_q=target_q,
            log_probs=log_probs
        )
        
        # Increment update counter first
        self.updates += 1
        
        # We need access to the run_id that's used elsewhere in the training code.
        # Since this is internal to the agent, we'll look for it as an attribute.
        run_id = getattr(self, 'run_id', None)
        
        # Save statistics periodically - after we've updated the counter
        if self.updates % self.save_stats_frequency == 0:
            self.save_learning_stats(run_id=run_id)
    
    def _track_learning_stats(self, states, actions, rewards, critic_loss, actor_loss, alpha_loss, q_values, target_q, log_probs):
        """Track learning statistics for later analysis."""
        # Convert tensors to numpy arrays for statistics calculation
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        rewards_np = rewards.detach().cpu().numpy().flatten()
        q_values_np = q_values.detach().cpu().numpy().flatten()
        target_q_np = target_q.detach().cpu().numpy().flatten()
        log_probs_np = log_probs.detach().cpu().numpy().flatten()
        
        # Calculate statistics
        self.learning_stats['step'].append(self.updates)
        
        # State statistics (for each dimension)
        state_mean = np.mean(states_np, axis=0)
        state_min = np.min(states_np, axis=0)
        state_max = np.max(states_np, axis=0)
        state_std = np.std(states_np, axis=0)
        # Kurtosis may fail with constant values, so use try-except
        state_kurtosis = np.zeros_like(state_mean)
        for i in range(len(state_mean)):
            try:
                state_kurtosis[i] = stats.kurtosis(states_np[:, i])
            except:
                state_kurtosis[i] = 0
        
        # For multi-dimensional state/action spaces, we save the mean across dimensions
        self.learning_stats['state_mean'].append(np.mean(state_mean))
        self.learning_stats['state_min'].append(np.mean(state_min))
        self.learning_stats['state_max'].append(np.mean(state_max))
        self.learning_stats['state_std'].append(np.mean(state_std))
        self.learning_stats['state_kurtosis'].append(np.mean(state_kurtosis))
        
        # Action statistics
        self.learning_stats['action_mean'].append(np.mean(actions_np))
        self.learning_stats['action_min'].append(np.min(actions_np))
        self.learning_stats['action_max'].append(np.max(actions_np))
        self.learning_stats['action_std'].append(np.std(actions_np))
        try:
            self.learning_stats['action_kurtosis'].append(stats.kurtosis(actions_np.flatten()))
        except:
            self.learning_stats['action_kurtosis'].append(0)
        
        # Reward statistics
        self.learning_stats['reward_mean'].append(np.mean(rewards_np))
        self.learning_stats['reward_min'].append(np.min(rewards_np))
        self.learning_stats['reward_max'].append(np.max(rewards_np))
        self.learning_stats['reward_std'].append(np.std(rewards_np))
        try:
            self.learning_stats['reward_kurtosis'].append(stats.kurtosis(rewards_np))
        except:
            self.learning_stats['reward_kurtosis'].append(0)
        
        # Network statistics
        self.learning_stats['critic_loss'].append(critic_loss.item())
        self.learning_stats['actor_loss'].append(actor_loss.item())
        self.learning_stats['alpha_loss'].append(alpha_loss.item())
        self.learning_stats['alpha_value'].append(self.alpha.item())
        
        # Q-value statistics
        self.learning_stats['q_value_mean'].append(np.mean(q_values_np))
        self.learning_stats['q_value_min'].append(np.min(q_values_np))
        self.learning_stats['q_value_max'].append(np.max(q_values_np))
        self.learning_stats['q_value_std'].append(np.std(q_values_np))
        
        # Target Q-value statistics
        self.learning_stats['target_q_mean'].append(np.mean(target_q_np))
        self.learning_stats['target_q_min'].append(np.min(target_q_np))
        self.learning_stats['target_q_max'].append(np.max(target_q_np))
        self.learning_stats['target_q_std'].append(np.std(target_q_np))
        
        # Log probability statistics
        self.learning_stats['log_prob_mean'].append(np.mean(log_probs_np))
        self.learning_stats['log_prob_min'].append(np.min(log_probs_np))
        self.learning_stats['log_prob_max'].append(np.max(log_probs_np))
        self.learning_stats['log_prob_std'].append(np.std(log_probs_np))
    
    def save_learning_stats(self, filename=None, run_id=None):
        """
        Save learning statistics to a CSV file.
        
        Args:
            filename: Optional explicit filename to save to
            run_id: Unique run identifier for consistent filenames across a training run
        """
        # Check if we have a run_dir attribute (set by the caller)
        run_dir = getattr(self, 'run_dir', None)
        
        if filename is None:
            if run_dir:
                # Use the run directory
                filename = f"{run_dir}/learning_stats/sac_learning_stats.csv"
            elif run_id is None:
                # Fallback to timestamp if no run_id provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/learning_stats/sac_learning_stats_{timestamp}.csv"
            else:
                # Use run_id for consistent filename
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/learning_stats/sac_learning_stats_{run_id}.csv"
        
        # Safety check - make sure we have stats to save
        if len(self.learning_stats['step']) == 0:
            print(f"No learning stats to save yet.")
            return
        
        # Instead of tracking the last saved step, we'll check if the file exists and read its content
        # to determine what steps have already been saved
        file_exists = os.path.isfile(filename)
        
        if file_exists:
            try:
                # Read existing file to get the last saved step
                existing_df = pd.read_csv(filename)
                if len(existing_df) > 0 and 'step' in existing_df.columns:
                    last_saved_step = existing_df['step'].max()
                else:
                    last_saved_step = -1  # No valid steps found, save everything
            except Exception as e:
                print(f"Error reading existing stats file: {e}")
                last_saved_step = -1  # On error, save everything
        else:
            last_saved_step = -1  # File doesn't exist, save everything
        
        # Find all steps that are newer than the last saved step
        steps = np.array(self.learning_stats['step'])
        new_indices = np.where(steps > last_saved_step)[0]
        
        if len(new_indices) > 0:
            # Extract all statistics for new steps
            latest_stats = {key: [self.learning_stats[key][i] for i in new_indices] for key in self.learning_stats}
            stats_df = pd.DataFrame(latest_stats)
            
            # Save to CSV with append mode if file exists
            if file_exists:
                stats_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                # Create new file with headers
                stats_df.to_csv(filename, index=False)
            
            
            # Store the last saved step for reference
            self.last_saved_step = int(max(steps[new_indices]))
            
            # Check if we should clear the stats to save memory during long training runs
            # Only clear if we've successfully saved and have a lot of accumulated stats
            if len(self.learning_stats['step']) > 5000:  # Arbitrary threshold
                self._clear_learning_stats_except_recent(keep_last=100)  # Keep the most recent entries
        else:
            print(f"No new learning stats to save since last save (last saved step: {last_saved_step}).")
    
    def _clear_learning_stats_except_recent(self, keep_last=100):
        """
        Clear most of the learning stats to prevent memory issues during long training runs,
        but keep the most recent entries for continuity.
        
        Args:
            keep_last: Number of most recent entries to keep
        """
        if len(self.learning_stats['step']) <= keep_last:
            return  # Nothing to clear
            
        # Keep only the most recent entries
        for key in self.learning_stats:
            self.learning_stats[key] = self.learning_stats[key][-keep_last:]
    
    def save(self, filename=None, run_id=None):
        """
        Save the model to disk.
        
        Args:
            filename: Optional explicit filename to save to
            run_id: Unique run identifier for consistent filenames across a training run
        """
        # Check if we have a run_dir attribute (set by the caller)
        run_dir = getattr(self, 'run_dir', None)
        
        if filename is None:
            if run_dir:
                # Use the run directory
                filename = f"{run_dir}/agent_saves/sac_pytorch"
            elif run_id is None:
                # Fallback to timestamp if no run_id provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/saves/sac_pytorch_{timestamp}"
            else:
                # Use run_id for consistent filename
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/saves/sac_pytorch_{run_id}"
        
        # Save learning statistics
        self.save_learning_stats(f"{filename}_learning_stats.csv", run_id)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'updates': self.updates
        }, filename)
        
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load the model from disk."""
        checkpoint = torch.load(filename)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.updates = checkpoint['updates']
        
        print(f"Model loaded from {filename}") 