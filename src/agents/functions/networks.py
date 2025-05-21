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
        sa = jnp.concatenate([state, action], axis=-1)

        # Q1 branch
        with nn.name_scope("q1"):
            x1 = sa
            for _ in range(self.number_of_hidden_layers):
                x1 = nn.Dense(self.hidden_dim)(x1)
                x1 = nn.relu(x1)
            q1 = nn.Dense(1)(x1)

        # Q2 branch
        with nn.name_scope("q2"):
            x2 = sa
            for _ in range(self.number_of_hidden_layers):
                x2 = nn.Dense(self.hidden_dim)(x2)
                x2 = nn.relu(x2)
            q2 = nn.Dense(1)(x2)

        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)
