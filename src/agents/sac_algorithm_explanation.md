# Soft Actor-Critic (SAC) Implementation in JAX

This document explains the implementation of the Soft Actor-Critic (SAC) algorithm in our codebase, the mathematical principles behind it, and how we leverage JAX for performance optimization.

## Table of Contents

1. [Introduction to SAC](#introduction-to-sac)
2. [Mathematical Framework](#mathematical-framework)
3. [Core Components](#core-components)
4. [JAX Implementation Details](#jax-implementation-details)
5. [Network Architecture](#network-architecture)
6. [Key Algorithm Techniques](#key-algorithm-techniques)
7. [Function Breakdown](#function-breakdown)
8. [Advanced Implementation Features](#advanced-implementation-features)
9. [Training Process](#training-process)
10. [Implementation Optimizations](#implementation-optimizations)
11. [Implementation-Specific Details](#implementation-specific-details)
12. [Application to Rocket Trajectory Optimization](#application-to-rocket-trajectory-optimization)
14. [Conclusion](#conclusion)

## Introduction to SAC

Soft Actor-Critic (SAC) is an off-policy maximum entropy deep reinforcement learning algorithm that provides sample-efficient learning while maintaining robust exploration. Unlike traditional RL algorithms that solely maximize expected return, SAC additionally optimizes for action entropy, encouraging exploration and preventing premature convergence to suboptimal policies.

Key advantages of SAC:
- **Sample efficiency**: Uses off-policy learning with a replay buffer
- **Stability**: Employs a temperature parameter to balance exploitation vs. exploration
- **Robustness**: Entropy maximization leads to better exploration and prevents policy collapse

## Mathematical Framework

### Objective Function

SAC aims to maximize both the expected return and the entropy:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(·|s_t)) \right]$$

Where:
- $r(s_t, a_t)$ is the reward
- $\alpha$ is the temperature parameter controlling the importance of entropy
- $\mathcal{H}(\pi(·|s_t))$ is the entropy of the policy at state $s_t$

### Gaussian Policy

We implement the policy as a Gaussian distribution where the actor network outputs the mean $\mu(s)$ and standard deviation $\sigma(s)$:

$$\pi(a|s) = \mathcal{N}(\mu(s), \sigma(s)^2)$$

The log-probability of an action under this policy is:

$$\log \pi(a|s) = -\frac{1}{2} \left( \frac{(a - \mu(s))^2}{\sigma(s)^2} + 2\log\sigma(s) + \log(2\pi) \right)$$

### Q-Function and Value Function

The soft Q-function and soft value function are defined as:

$$Q(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}}[V(s_{t+1})]$$

$$V(s_t) = \mathbb{E}_{a_t \sim \pi} [Q(s_t, a_t) - \alpha \log \pi(a_t|s_t)]$$

### Temporal Difference (TD) Error

The TD error measures the difference between the current Q-value estimate and the target:

$$\delta = r_t + \gamma (1 - d_t) (Q_{target}(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1}|s_{t+1})) - Q(s_t, a_t)$$

Where $d_t$ is a binary done signal.

## Core Components

Our implementation consists of three key components:

1. **Critic (Q-function)**: Two Q-networks (for variance reduction) that estimate the action-value function
2. **Actor (Policy)**: A network that outputs the mean and standard deviation of a Gaussian policy
3. **Temperature parameter**: Automatically adjusted to maintain a target entropy level

## JAX Implementation Details

Our implementation leverages JAX for high-performance, hardware-accelerated operations:

### Key JAX Features Used

1. **JIT Compilation**: We use `@jax.jit` to just-in-time compile functions, dramatically improving execution speed
2. **Automatic Differentiation**: JAX's gradient capabilities (`jax.grad`) are used for policy and value function optimization
3. **Functional Updates**: We follow JAX's functional programming model, avoiding in-place updates
4. **Vectorization**: We use JAX's vectorized operations for batch processing
5. **Tree Operations**: JAX's tree utilities help manipulate nested parameter structures

### Functional Programming Paradigm

JAX enforces a functional programming style, meaning:
- No in-place modifications of arrays
- Parameters are explicitly passed and returned
- Pure functions with no side effects

### JAX-Specific Optimizations

1. **Compilation with XLA**: Our functions are JIT-compiled using XLA (Accelerated Linear Algebra), which optimizes the computation graph before execution.

   ```python
   @jax.jit
   def gaussian_likelihood(actions, mean, std):
       # Function body...
   ```

2. **Partial Function Application**: We use `functools.partial` to create specialized versions of functions with fixed parameters:

   ```python
   critic_update_lambda = jax.jit(
       partial(critic_update,
               critic_optimiser=critic_optimiser,
               calculate_td_error_fcn=calculate_td_error_lambda,
               critic_grad_max_norm=critic_grad_max_norm),
       static_argnames=['critic_optimiser', 'calculate_td_error_fcn', 'critic_grad_max_norm']
   )
   ```

3. **Static Argument Annotations**: We use `static_argnames` to inform JAX which arguments should be treated as static during compilation.

4. **Tree Manipulation**: We use JAX's tree utilities for handling neural network parameters:

   ```python
   critic_target_params = jax.tree_util.tree_map(
       lambda p, tp: tau * p + (1.0 - tau) * tp, 
       critic_params, 
       critic_target_params
   )
   ```

5. **Conditional Computation**: We use `jax.lax.cond` for efficient conditional operations without breaking JIT compilation:

   ```python
   temperature, temperature_opt_state, temperature_loss = jax.lax.cond(
       first_step_bool,
       lambda: (temperature, temperature_opt_state, 0.0),
       lambda: temperature_update_lambda(...)
   )
   ```

## Network Architecture

Our SAC implementation uses two types of neural networks: the actor and the critic.

### Actor Network

The actor network maps states to action distributions. It outputs both the mean and standard deviation of a Gaussian distribution:

```python
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int
    number_of_hidden_layers: int
    
    @nn.compact
    def __call__(self, x):
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Hidden layers
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        
        # Output layers
        action_mean = nn.Dense(self.action_dim)(x)
        
        # Standard deviation with softplus to ensure positivity
        log_std = nn.Dense(self.action_dim)(x)
        action_std = nn.softplus(log_std)
        
        return action_mean, action_std
```

### Critic Network (Double Q-Network)

The critic network implements two Q-functions to reduce overestimation bias:

```python
class DoubleCritic(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int
    number_of_hidden_layers: int
    
    @nn.compact
    def __call__(self, states, actions):
        sa = jnp.concatenate([states, actions], axis=-1)
        
        # First Q-network
        q1 = nn.Dense(self.hidden_dim)(sa)
        q1 = nn.relu(q1)
        
        for _ in range(self.number_of_hidden_layers):
            q1 = nn.Dense(self.hidden_dim)(q1)
            q1 = nn.relu(q1)
        
        q1 = nn.Dense(1)(q1)
        
        # Second Q-network (separate parameters)
        q2 = nn.Dense(self.hidden_dim)(sa)
        q2 = nn.relu(q2)
        
        for _ in range(self.number_of_hidden_layers):
            q2 = nn.Dense(self.hidden_dim)(q2)
            q2 = nn.relu(q2)
        
        q2 = nn.Dense(1)(q2)
        
        return q1, q2
```

## Key Algorithm Techniques

### Reparameterization Trick

The reparameterization trick is crucial for backpropagating through stochastic actions. Instead of directly sampling from the policy distribution (which would break the gradient flow), we parameterize actions as:

$$a = \mu(s) + \sigma(s) \cdot \epsilon$$

where $\epsilon \sim \mathcal{N}(0, 1)$ is a random noise variable.

In our implementation:

```python
# In actor_update
def loss_fcn(params):
    action_mean, action_std = actor.apply(params, states)
    actions = normal_distribution * action_std + action_mean
    # Continue with Q-value computation...
```

This allows gradients to flow through the action mean and standard deviation, even though the actions themselves are stochastic.

### Automatic Temperature Tuning

The temperature parameter $\alpha$ controls the trade-off between entropy maximization (exploration) and reward maximization (exploitation). Rather than manually tuning this parameter, we adjust it automatically to achieve a target entropy level:

```python
def temperature_update(temperature_optimiser,
                       temperature_grad_max_norm: float,
                       current_log_probabilities: jnp.ndarray,
                       target_entropy: float,
                       temperature_opt_state: jnp.ndarray,
                       temperature: float):
    """Update log_alpha so that E[−log π] ≈ target_entropy."""
    log_alpha = jnp.log(temperature)
    def loss_fn(log_alpha):
        diff = jax.lax.stop_gradient(current_log_probabilities + target_entropy)
        return - (log_alpha * diff).mean()

    grads = jax.grad(loss_fn)(log_alpha)
    # Apply updates...
    temperature = jnp.exp(log_alpha)
    return temperature, temperature_opt_state, temperature_loss
```

The target entropy is typically set to `-action_dim`, meaning the policy should have an entropy equivalent to a uniform distribution over actions. When policy entropy is too low (under-exploration), $\alpha$ increases; when it's too high (over-exploration), $\alpha$ decreases.

This mechanism allows the agent to automatically find the right balance between exploration and exploitation throughout training, adapting as the policy improves.

## Function Breakdown

Let's examine the key functions in our implementation:

### `gaussian_likelihood`

```python
@jax.jit
def gaussian_likelihood(actions: jnp.ndarray,
                        mean: jnp.ndarray,
                        std: jnp.ndarray) -> jnp.ndarray:
    log_prob = -0.5 * (
        ((actions - mean) ** 2) / (std ** 2)  # Quadratic term
        + 2 * jnp.log(std)  # Log scale normalization
        + jnp.log(2 * jnp.pi)  # Constant factor
    )
    return log_prob.sum(axis=-1)  # Sum over the action dimensions
```

This function calculates the log probability of actions under a Gaussian distribution. It's used to compute the policy entropy term in the SAC objective.

### `calculate_td_error`

```python
def calculate_td_error(states,
                       actions,
                       rewards,
                       next_states,
                       dones,
                       temperature: float,
                       gamma: float,
                       critic_params: jnp.ndarray,
                       critic_target_params: jnp.ndarray,
                       critic: nn.Module,
                       next_actions: jnp.ndarray,
                       next_log_policy: jnp.ndarray) -> jnp.ndarray:
    q1, q2 = critic.apply(critic_params, states, actions)
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, next_actions)
    next_q_mean = jnp.minimum(next_q1, next_q2)
    entropy_term = temperature * jnp.expand_dims(next_log_policy, axis=1)  
    td_target = rewards + gamma * (1 - dones) * (next_q_mean - entropy_term)
    td_errors = 0.5 * ((td_target - q1)**2 + (td_target - q2)**2)
    return td_errors
```

This function computes the TD errors for the critic update. It:
1. Gets Q-values for current state-action pairs
2. Gets target Q-values for next state-action pairs
3. Takes the minimum Q-value for variance reduction
4. Adds the entropy term weighted by temperature
5. Computes the TD target with rewards and discounting
6. Calculates squared errors for both Q-networks

### `critic_update`

This function updates the critic networks using the TD errors:
1. Computes the loss function with importance sampling weights from the prioritised replay buffer
2. Calculates gradients with respect to critic parameters
3. Clips gradients to prevent exploding gradients
4. Applies updates using the optimizer

### `actor_update`

Updates the actor network to maximize expected return plus entropy:
1. Gets action means and standard deviations from the actor
2. Samples actions using the reparameterization trick
3. Evaluates these actions with the critic
4. Combines Q-values with log probabilities to compute the actor loss
5. Updates actor parameters using gradients

### `temperature_update`

Automatically adjusts the temperature parameter to maintain a target entropy:
1. Computes the difference between current policy entropy and target entropy
2. Updates temperature to minimize this difference

### `update_sac`

Orchestrates the complete SAC update:
1. Samples next actions for target computation
2. Updates the critic using TD errors
3. Updates the actor to maximize Q-values minus entropy
4. Updates the temperature parameter
5. Performs soft updates to target networks
6. Returns updated parameters and metrics

## Advanced Implementation Features

### Gradient Clipping

Gradient clipping is a crucial technique to prevent exploding gradients, ensuring training stability. Our implementation uses the `clip_grads` function:

```python
@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads
```

This function:
1. Calculates the global norm of the gradient across all parameters
2. Computes a scaling factor that caps the norm at `max_norm`
3. Applies this scaling uniformly to all gradient components
4. Handles nested parameter structures using JAX's tree utilities

Each network (actor, critic, temperature) has its own `max_norm` hyperparameter, allowing for fine-grained control over gradient magnitudes.

### Prioritized Experience Replay (PER) Integration

Our implementation incorporates Prioritized Experience Replay, which samples transitions with higher TD errors more frequently. The PER mechanism integrates with SAC through:

1. **Importance Sampling Weights**: The critic loss uses weights from the buffer to correct for the sampling bias:

   ```python
   weighted_td_error_loss = jnp.mean(jax.lax.stop_gradient(buffer_weights) * td_errors)
   ```

2. **Priority Updates**: After computing new TD errors, we update the transitions' priorities:

   ```python
   self.buffer.update_priorities(index, td_errors)
   ```

3. **Buffer Initialization**: The buffer uses parameters α (priority exponent) and β (importance sampling correction):

   ```python
   self.buffer = PERBuffer(
       gamma=gamma,
       alpha=alpha_buffer,  # Controls how much prioritization is used
       beta=beta_buffer,    # Corrects for importance sampling bias
       beta_decay=beta_decay_buffer,  # Anneals β toward 1
       # ...other parameters...
   )
   ```

The PER implementation helps focus learning on the most informative transitions, significantly improving sample efficiency while maintaining unbiased updates through importance sampling correction.

## Training Process

The complete training process follows these steps:

1. **Initialization**:
   - Initialize actor and critic networks
   - Set initial temperature
   - Create target networks as copies of critics

2. **Data Collection**:
   - Sample actions from the current policy with added noise
   - Store transitions in a prioritised replay buffer

3. **Optimization** (for each batch):
   - Sample a mini-batch of transitions with priorities
   - Update critics to minimize TD error
   - Update actor to maximize Q-value and entropy
   - Adjust temperature parameter
   - Soft-update target networks
   - Update priorities in the replay buffer

4. **Repeat** until convergence

## Implementation Optimizations

Our implementation includes several optimizations:

1. **Double Q-Learning**: Two Q-networks to reduce overestimation bias
2. **Target Networks**: Slowly updated copies of critics for stability
3. **Prioritized Experience Replay**: Samples important transitions more frequently
4. **Gradient Clipping**: Prevents exploding gradients
5. **Automatic Temperature Tuning**: Adjusts exploration vs. exploitation
6. **JIT Compilation**: Accelerates computation with XLA

## Implementation-Specific Details

Our specific implementation has several key characteristics that distinguish it from standard SAC implementations:

### Custom Actor Activations

Our actor network implementation uses a combination of tanh and sigmoid activations:

```python
class Actor(nn.Module):
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
        std = nn.sigmoid(nn.Dense(self.action_dim)(x))
        return mean, std
```

- The tanh activation for action mean bounds outputs to [-1, 1], which is appropriate for normalized continuous control environments
- Using sigmoid instead of softplus for the standard deviation bounds it to [0, 1], providing better control over exploration

### Configurable Network Size

Both actor and critic networks can be easily configured with different depths and widths:

```python
self.actor = Actor(action_dim=action_dim,
                  hidden_dim=hidden_dim_actor,
                  number_of_hidden_layers=number_of_hidden_layers_actor)
                  
self.critic = DoubleCritic(state_dim=state_dim,
                          action_dim=action_dim,
                          hidden_dim=hidden_dim_critic,
                          number_of_hidden_layers=number_of_hidden_layers_critic)
```

This allows us to adjust network capacity based on task complexity:
- Larger, deeper networks for complex dynamics (e.g., supersonic flight phases)
- Smaller networks for simpler tasks (e.g., hover control)

### Critic Warm-Up Phase

We incorporate a specific warm-up phase for the critic before joint actor-critic training:

```python
def critic_warm_up_step(self):
    states, actions, rewards, next_states, dones, _, _ = self.buffer(self.get_subkey())
    self.critic_params, self.critic_opt_state, self.critic_target_params, critic_loss_warm_up = self.critic_warm_up_update_lambda(
        actor_params=self.actor_params,
        normal_distribution_for_next_actions=self.get_normal_distributions_batched(),
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        critic_params=self.critic_params,
        critic_target_params=self.critic_target_params,
        critic_opt_state=self.critic_opt_state
    )
    # Logging and tracking
    self.writer.add_scalar('CriticWarmUp/Loss', np.array(critic_loss_warm_up), self.critic_warm_up_step_idx)
    self.critic_warm_up_step_idx += 1
    return critic_loss_warm_up
```

This warm-up enhances stability by:
1. Pre-training the critic to better approximate Q-values before policy updates begin
2. Reducing initial actor updates based on poorly estimated Q-functions
3. Allowing the prioritised replay buffer to develop meaningful priorities before full training

### Parameterized Target Entropy

Unlike standard implementations that set target entropy to `-action_dim`, we parameterize it using max standard deviation:

```python
self.target_entropy = -self.action_dim * jnp.log(self.max_std)
```

This allows us to adjust the target entropy based on the desired exploration level for each task.

## Application to Rocket Trajectory Optimization

Our SAC implementation is specifically tailored for rocket trajectory optimization, which presents unique challenges:

### Flight Phase-Specific Optimization

The rocket control problem is divided into distinct flight phases, each with its own dynamics and control objectives:

1. **Supersonic/Subsonic Ascent**: Controlling aerodynamic surfaces during full rocket ascent
2. **Flip-Over Maneuver & Boostback Burn**: Rapidly reorienting the rocket to prepare for landing, and engines fire to cancel out horizontal velocity
3. **Ballistic Arc**: Ensures an aerodynamically stable rocket.
4. **Landing Burn**: Final precision control for soft touchdown

Each phase uses a separate SAC agent with hyperparameters optimized for that specific phase's dynamics:

```python
config_flip_over_boostbackburn = {
    'sac': {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 50,
        'number_of_hidden_layers_critic': 18,
        'temperature_initial': 0.005,
        'gamma': 0.98,
        'tau': 0.0005,
        # PER parameters
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        # Other parameters
        'buffer_size': 500000,
        'trajectory_length': 200,
        'batch_size': 2048,
        # Learning rates
        'critic_learning_rate': 1e-4,
        'actor_learning_rate': 5e-12,
        'temperature_learning_rate': 1e-7,
        # Gradient clipping
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.1,
        'max_std': 0.001
    },
    'num_episodes': 1500000,
    'critic_warm_up_steps': 1000,
    'pre_train_critic_learning_rate': 2e-4,
    'pre_train_critic_batch_size': 256
}
```

### High-Dimensional, Continuous Control

Rocket control requires precise high-dimensional continuous control with complex non-linear dynamics:

- **State Space**: Includes position, velocity, attitude, angular velocity, mass, aerodynamic parameters
- **Action Space**: Includes engine throttle, gimbal angles, grid fin deflections, RCS thruster controls
- **Time-Varying Dynamics**: Aerodynamic properties, mass, and control authority change dramatically throughout flight

Our SAC implementation handles these challenges through:
1. Deep networks for function approximation
2. Automatic entropy adjustment to balance exploration/exploitation
3. Prioritized replay to focus on challenging flight regimes

### Safety Constraints

Rocket control requires maintaining multiple safety constraints:

- Structural load limitations (dynamic pressure)
- Propellant margins
- Landing precision requirements

We incorporate these through carefully designed reward functions that strongly penalize constraint violations.


## Conclusion

Our JAX implementation of SAC provides a highly efficient, mathematically sound reinforcement learning algorithm for rocket trajectory optimization. The combination of entropy-regularized RL with JAX's performance optimizations results in a system that learns complex behaviors efficiently while maintaining robustness to hyperparameters. 