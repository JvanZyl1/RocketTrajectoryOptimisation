# SAC Functions Verification Documentation

This document provides verification and analysis for the utility functions used in our Soft Actor-Critic (SAC) implementation. It focuses on critical computational functions that are essential for the algorithm's stability and performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Gaussian Likelihood Function](#gaussian-likelihood-function)
   - [Mathematical Foundation](#mathematical-foundation)
   - [Implementation](#implementation)
   - [Verification Tests](#verification-tests)
3. [Gradient Clipping Function](#gradient-clipping-function)
   - [Mathematical Foundation](#mathematical-foundation-1)
   - [Implementation](#implementation-1)
   - [Verification Tests](#verification-tests-1)
4. [TD Error Calculation Function](#td-error-calculation-function)
   - [Mathematical Foundation](#mathematical-foundation-2)
   - [Implementation](#implementation-2)
   - [Verification Tests](#verification-tests-2)
5. [Running the Verification Tests](#running-the-verification-tests)
6. [Conclusion](#conclusion)

## Introduction

Soft Actor-Critic (SAC) is a state-of-the-art reinforcement learning algorithm that optimizes a stochastic policy in an off-policy way. It combines several key ideas including:

1. An actor-critic architecture with separate policy and value function networks
2. A maximum entropy framework that encourages exploration
3. Off-policy learning from a replay buffer

The implementation of SAC relies on several utility functions that must work correctly for the algorithm to perform well. This document focuses on verifying three critical functions:

1. `gaussian_likelihood`: Computes the log probability density of actions under the policy's Gaussian distribution
2. `clip_grads`: Implements gradient clipping to maintain training stability
3. `calculate_td_error`: Computes the temporal difference (TD) error used for training the critic and prioritizing experiences

## Gaussian Likelihood Function

### Mathematical Foundation

In SAC, the policy is modeled as a Gaussian distribution with a learned mean and standard deviation. For a given action `a`, mean `μ` and standard deviation `σ`, the log probability density is:

$$\log p(a|\mu,\sigma) = -\frac{1}{2}\left(\frac{(a - \mu)^2}{\sigma^2} + 2\log\sigma + \log(2\pi)\right)$$

For multi-dimensional actions, we sum the log probabilities across dimensions, assuming independence:

$$\log p(\mathbf{a}|\mathbf{\mu},\mathbf{\sigma}) = \sum_{i=1}^{n} \log p(a_i|\mu_i,\sigma_i)$$

This function is critical for:
- Computing the policy's entropy for the entropy regularization term
- Evaluating the policy during Q-value updates
- Computing importance weights when needed

### Implementation

The `gaussian_likelihood` function implements this mathematical formula using JAX for efficient computation:

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

The function is JIT-compiled for performance and operates on JAX arrays.

### Verification Tests

We verify the correctness of the `gaussian_likelihood` function through several tests:

1. **Simple 1D Case at Mean**: Testing with action=mean=0, std=1, which should produce a log probability of -0.5*log(2π)
2. **1D Case Away from Mean**: Testing with action=1, mean=0, std=1, which adds a quadratic penalty term
3. **Multidimensional Case**: Testing with various combinations of actions, means, and standard deviations
4. **Visual Verification**: Generating a plot to visualize the log probability across a range of values

These tests ensure that the function correctly implements the mathematical formula and handles different input configurations properly.

## Gradient Clipping Function

### Mathematical Foundation

Gradient clipping is an important technique for stabilizing neural network training, especially in reinforcement learning where gradients can have high variance. The technique limits the norm of the gradient vector to a maximum value, which prevents excessively large parameter updates.

For a gradient vector `g` and a maximum norm `max_norm`, the clipped gradient is:

$$\text{clip}(g) = \begin{cases}
g & \text{if } \|g\| \leq \text{max\_norm} \\
\frac{\text{max\_norm}}{\|g\|} g & \text{otherwise}
\end{cases}$$

where $\|g\|$ is the L2 norm of the gradient vector.

### Implementation

The `clip_grads` function implements gradient clipping using JAX's tree manipulation functions to handle nested parameter structures:

```python
@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads
```

The function:
1. Computes the L2 norm of all gradient values across the parameter tree
2. Determines a scaling factor to apply to all gradients (1.0 if norm is below threshold)
3. Applies the scaling factor to all gradients
4. Returns the clipped gradients with the same structure as the input

### Verification Tests

We verify the correctness of the `clip_grads` function through several tests:

1. **Gradient Below Threshold**: Testing with gradients whose norm is below the maximum allowed value, which should remain unchanged
2. **Gradient Above Threshold**: Testing with gradients whose norm exceeds the maximum, which should be scaled down
3. **Structured Gradients**: Testing with gradients in a nested dictionary structure (similar to neural network parameters), ensuring the function handles complex parameter structures correctly

## TD Error Calculation Function

### Mathematical Foundation

The Temporal Difference (TD) error is central to reinforcement learning algorithms. In SAC, the TD error is computed using the following steps:

1. Get the current Q-values (Q1, Q2) for the state-action pair from the critic network
2. Get the next Q-values for the next state and next action from the target critic network
3. Take the minimum of the two next Q-values to reduce overestimation bias
4. Subtract the entropy term (temperature * log_probability) to incorporate the maximum entropy objective
5. Compute the TD target as reward + gamma * (1 - done) * (min_next_Q - entropy_term)
6. Compute the TD error as the squared difference between the TD target and current Q-values

Mathematically, for a transition (s, a, r, s', a'):

$$Q_{target} = r + \gamma \cdot (1 - done) \cdot (\min(Q_1(s', a'), Q_2(s', a')) - \alpha \cdot \log \pi(a'|s'))$$

$$TD_{error} = \frac{1}{2}[(Q_{target} - Q_1(s, a))^2 + (Q_{target} - Q_2(s, a))^2]$$

where:
- $\alpha$ is the temperature parameter controlling the entropy regularization
- $\pi(a'|s')$ is the policy probability of taking action $a'$ in state $s'$
- $done$ is a binary flag indicating whether the state $s'$ is terminal

### Implementation

The `calculate_td_error` function implements this TD error calculation:

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

This function uses JAX for efficient computation and handles batched inputs for experience replay.

### Verification Tests

We verify the correctness of the `calculate_td_error` function through several tests:

1. **Basic TD Error Calculation**: Testing TD error calculation for non-terminal states with different rewards
2. **Terminal State Handling**: Testing that the function correctly zeroes out future rewards for terminal states
3. **Temperature Effect**: Testing how different temperature values affect the TD error calculation
4. **Visual Verification**: Generating a plot showing how TD error varies with different reward values

These tests ensure that the function correctly implements the TD error calculation for both terminal and non-terminal states and properly incorporates the entropy regularization term.

## Running the Verification Tests

The verification tests for these functions are implemented in `src/agents/functions/verification/sac_functions_verification.py`. To run the tests:

```bash
python src/agents/functions/verification/sac_functions_verification.py
```

The tests will:
1. Execute all verification checks for all three functions
2. Generate visualization plots in the `results/verification/sac_functions_verification/` directory
3. Save a CSV file with the test results
4. Display a summary of passing and failing tests

## Conclusion

The verification tests confirm that the `gaussian_likelihood`, `clip_grads`, and `calculate_td_error` functions in our SAC implementation correctly implement their respective mathematical formulas and handle all expected input configurations.

These functions are critical to the stability and performance of the SAC algorithm:

- `gaussian_likelihood` correctly computes log probabilities for the policy's actions, enabling proper entropy regularization and importance weighting
- `clip_grads` correctly implements gradient clipping, preventing parameter updates from becoming too large and destabilizing training
- `calculate_td_error` correctly computes the TD errors that drive the learning process, ensuring proper handling of terminal states and entropy regularization

By verifying these functions, we can have confidence in the mathematical foundations of our SAC implementation, which contributes to its overall reliability and performance. 