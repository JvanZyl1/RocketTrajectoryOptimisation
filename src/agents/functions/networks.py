import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

### ACTOR ###
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        std = nn.softplus(nn.Dense(self.action_dim)(x))
        return mean, std
    
### CRITIC ###
class DoubleCritic(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([state, action], axis=-1)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean_q1 = nn.Dense(1)(x)
        mean_q2 = nn.Dense(1)(x)
        return mean_q1, mean_q2
    

### TENSORFLOW COPIES FOR LOGGING ###
# Don't automatically update the activation functions btw.
import tensorflow as tf

class ActorTF(tf.keras.Model):
    def __init__(self,
                 flax_actor: Actor,
                 state_dim: int):
        super(ActorTF, self).__init__()


        self.hidden_layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(flax_actor.hidden_dim, activation='relu', name=f'hidden_layer_{i+1}')
             for i in range(flax_actor.number_of_hidden_layers)]
        )

        # Separate output layers for mean and std
        self.mean_layer = tf.keras.layers.Dense(flax_actor.action_dim, activation='tanh', name='mean_layer')
        self.std_layer = tf.keras.layers.Dense(flax_actor.action_dim, activation='softplus', name='std_layer')

    def call(self, inputs):
        x = self.hidden_layers(inputs)
        mean = self.mean_layer(x)
        std = self.std_layer(x)
        return mean, std
    
class DoubleCriticTF(tf.keras.Model):
    def __init__(self, flax_critic: DoubleCritic, state_dim: int, action_dim: int):
        super(DoubleCriticTF, self).__init__()


        # Hidden layers
        self.hidden_layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(flax_critic.hidden_dim, activation='relu',
                                   name=f'hidden_layer_{i+1}')
             for i in range(flax_critic.number_of_hidden_layers)]
        )

        # Separate output layers for Q1 and Q2
        self.q1_output = tf.keras.layers.Dense(1, name='Q1_output')
        self.q2_output = tf.keras.layers.Dense(1, name='Q2_output')

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.hidden_layers(x)
        q1 = self.q1_output(x)
        q2 = self.q2_output(x)
        return q1, q2

### GRAPH WRITER ###

def write_graph(writer,
                flax_actor : Actor,
                flax_critic : DoubleCritic,
                state_dim : int,
                action_dim : int):
    
    actor_tf = ActorTF(flax_actor, state_dim)
    critic_tf = DoubleCriticTF(flax_critic, state_dim, action_dim)

    tf.summary.trace_on(graph=True, profiler=False)

    dummy_state = tf.random.normal((1, state_dim), name="state_input")
    dummy_action = tf.random.normal((1, action_dim), name="action_input")

    _ = actor_tf(dummy_state)
    _ = critic_tf(dummy_state, dummy_action)

    # Export Actor Graph
    with writer:
        tf.summary.trace_export(name="Actor_Graph", step=0)

    # Export Critic Graph
    with writer:
        tf.summary.trace_export(name="Critic_Graph", step=0)

    writer.flush()

    print(f"TensorBoard network graphs saved.")
        