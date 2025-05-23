import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

### ACTOR ###
class GaussianActor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        std = nn.sigmoid(nn.Dense(self.action_dim, kernel_init= nn.initializers.xavier_uniform(),
                        bias_init=nn.initializers.constant(0.0001))(x))
        return mean, std
    
class ClassicalActor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))

### CRITIC ###
class DoubleCritic(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # squish actions from (1,1,4) to (1,4)        
        x = jnp.concatenate([state, action], axis=-1)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean_q1 = nn.Dense(1)(x)
        mean_q2 = nn.Dense(1)(x)
        return mean_q1, mean_q2
        
class ValueNetwork(nn.Module):
    hidden_dim: int = 256
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        value = nn.Dense(1)(x)
        return value