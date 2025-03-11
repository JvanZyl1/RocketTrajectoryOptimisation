import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple

class Actor(nn.Module):
    """
    Policy network (Actor) implementation.

    Attributes:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        hidden_dim: Dimension of hidden layers (default: 256).
        stochastic: Whether the policy is stochastic or deterministic (default: True).
    """
    action_dim: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of the actor network.
        
        Args:
            state: Input state tensor of shape (state_dim).
        
        Returns:
            mean: Mean of the action distribution [jnp.ndarray].
            log_std: Log standard deviation of the action distribution [jnp.ndarray].
        """
        # Hidden layers
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)

        # Output mean, using tanh for [-1, 1] range
        mean = nn.Dense(self.action_dim)(x)

        # Output std, using softplus to ensure positivity
        std = nn.Dense(self.action_dim)(x)
        std = nn.softplus(std)
        return mean, std
    
class DoubleCritic(nn.Module):
    """
    Gaussian critic module.

    Attributes:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        hidden_dim: Dimension of the hidden layers (default: 256).
        activation_fn: Activation function (default: nn.relu).

    Returns:
        q1: First Q-value.
        q2: Second Q-value.
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    activation_fn: callable = nn.relu

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if action.ndim == 1:
            action = jnp.expand_dims(action, axis=1)
        if state.ndim == 3:
            state = jnp.squeeze(state, axis=0)
        # Concatenate state and action along the last dimension
        x = jnp.concatenate([state, action], axis=-1)
        
        # Define the shared hidden layers
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation_fn(x)

        # Gaussian outputs for Q1
        mean_q1 = nn.Dense(1)(x)

        # Gaussian outputs for Q2
        mean_q2 = nn.Dense(1)(x)

        return mean_q1, mean_q2
