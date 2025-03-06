import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple

jax.clear_caches()

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
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        mean = nn.tanh(mean)

        # Output std, using softplus to ensure positivity
        std = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        # Ensure between 0 and 1 by using sigmoid
        std_0_1 = nn.sigmoid(std)
        
        return mean, std_0_1


# C51 critic network
class DoubleDistributionalCritic(nn.Module):
    """
    Double distributional critic module.

    Attributes:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        hidden_dim: Dimension of the hidden layers (default: 256).
        num_points: Number of points in the distribution (default: 51).
        v_min: Minimum value for the critic distribution (default: -10.0).
        v_max: Maximum value for the critic distribution (default: 10.0).
        activation_fn: Activation function (default: nn.relu).

    Returns:
        q1: First distributional Q-value (logits over points).
        q2: Second distributional Q-value (logits over points).
        z: Support of the distribution.
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_points: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    activation_fn: callable = nn.relu

    def setup(self):
        # Define support of the distribution as a static member
        self.z = jnp.linspace(self.v_min, self.v_max, self.num_points)

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of the double distributional critic network.

        Args:
            state: Input state tensor of shape (state_dim).
            action: Input action tensor of shape (action_dim).

        Returns:
            q1: First distributional Q-value logits of shape (num_points).
            q2: Second distributional Q-value logits of shape (num_points).
            z: Support of the distribution (static).
        """
        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)

        # First critic network (Q1)
        q1 = nn.Dense(self.hidden_dim)(x)
        q1 = self.activation_fn(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = self.activation_fn(q1)
        q1 = nn.Dense(self.num_points)(q1)  # Output distribution logits for Q1

        # Second critic network (Q2)
        q2 = nn.Dense(self.hidden_dim)(x)
        q2 = self.activation_fn(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = self.activation_fn(q2)
        q2 = nn.Dense(self.num_points)(q2)  # Output distribution logits for Q2

        # Return both Q-value distributions and support
        return q1, q2, self.z
    
# Gaussian critic: atm a non-distributional critic is used.
class GaussianDoubleCritic(nn.Module):
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
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = self.activation_fn(x)

        # Gaussian outputs for Q1
        mean_q1 = nn.Dense(1,
                           kernel_init=nn.initializers.constant(0.0),  # Initialize weights to zero
                           bias_init=nn.initializers.constant(0.0))(x)

        # Gaussian outputs for Q2
        mean_q2 = nn.Dense(1,
                           kernel_init=nn.initializers.constant(0.0),  # Initialize weights to zero
                           bias_init=nn.initializers.constant(0.0))(x)

        return mean_q1, mean_q2
