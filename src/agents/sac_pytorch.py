import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
import os
from datetime import datetime

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
        flight_phase: str = "default"
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
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update of the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        
        self.updates += 1
    
    def save(self, filename=None):
        """Save the model to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/saves/sac_pytorch_{timestamp}"
        
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