# Actor Update Function Verification

This document provides verification and analysis for the `actor_update` function, which is a core component of our Soft Actor-Critic (SAC) implementation. It focuses on how the actor policy is updated during training.

## Table of Contents

1. [Introduction](#introduction)
2. [The Actor Update Function](#the-actor-update-function)
    - [Mathematical Foundation](#mathematical-foundation)
    - [Implementation](#implementation)
    - [Key Aspects](#key-aspects)
3. [Verification Tests](#verification-tests)
    - [Basic Functionality Test](#basic-functionality-test)
    - [Gradient Clipping Test](#gradient-clipping-test)
    - [Temperature Test](#temperature-test)
    - [Temperature Entropy Test](#temperature-entropy-test)
4. [Expected Test Results](#expected-test-results)
5. [Running the Verification Tests](#running-the-verification-tests)
6. [Conclusion](#conclusion)

## Introduction

In Soft Actor-Critic (SAC), the actor network represents a stochastic policy that maps states to distributions over actions. The actor is updated to maximize expected future returns while maintaining appropriate entropy for exploration. The `actor_update` function is responsible for:

1. Computing the policy outputs (action means and standard deviations)
2. Sampling actions from the policy distribution
3. Estimating Q-values for these actions using the critic
4. Computing the policy loss as a combination of Q-values and entropy
5. Applying gradient clipping to improve training stability
6. Updating the actor parameters using an optimizer
7. Returning the updated parameters and relevant metrics

The actor update is a critical component of SAC since it directly controls the learned policy that will be used for agent behavior.

## The Actor Update Function

### Mathematical Foundation

The actor update in SAC follows the policy improvement principle, but with an additional entropy term. For a batch of states, the actor computes a stochastic policy π(a|s) that outputs action means and standard deviations, and is trained to maximize:

$$J(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi} \left[ \min_{i=1,2} Q_{\theta_i}(s, a) - \alpha \log \pi_\phi(a|s) \right]$$

Where:
- $\phi$ represents the actor network parameters
- $\theta_i$ represents the critic network parameters
- $\mathcal{D}$ is the replay buffer
- $\alpha$ is the temperature parameter (controlling exploration)
- $\pi_\phi(a|s)$ is the policy (probability of action $a$ given state $s$)
- $Q_{\theta_i}(s, a)$ are the Q-values from both critics

The objective balances two competing goals:
1. Maximize the expected Q-value (exploitation)
2. Maximize policy entropy (exploration)

### Implementation

The `actor_update` function implements this mathematical foundation:

```python
def actor_update(actor_optimiser,
                 actor : nn.Module,
                 critic : nn.Module,
                 actor_grad_max_norm : float,
                 temperature : float,
                 states : jnp.ndarray,
                 normal_distribution : jnp.ndarray,
                 critic_params : jnp.ndarray,
                 actor_params : jnp.ndarray,
                 actor_opt_state : jnp.ndarray):
    def loss_fcn(params):
        action_mean, action_std = actor.apply(params, jax.lax.stop_gradient(states))
        actions = jax.lax.stop_gradient(normal_distribution) * action_std + action_mean
        action_std = jnp.maximum(action_std, 1e-6) # avoid crazy log probabilities.
        q1, q2 = critic.apply(jax.lax.stop_gradient(critic_params), jax.lax.stop_gradient(states), actions)
        q_min = jnp.minimum(q1, q2)
        log_probability = gaussian_likelihood(actions, action_mean, action_std)
        return (temperature * log_probability - q_min).mean(), (log_probability, action_std)
    grads, aux_values = jax.grad(loss_fcn, has_aux=True)(actor_params)
    # The aux_values variable is a tuple containing (log_probability, action_std)
    clipped_grads = clip_grads(grads, max_norm=actor_grad_max_norm)
    updates, actor_opt_state = actor_optimiser.update(clipped_grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    actor_loss, (current_log_probabilities, action_std) = loss_fcn(actor_params)
    return actor_params, actor_opt_state, actor_loss, current_log_probabilities, action_std
```

### Key Aspects

1. **Stochastic Policy**: The actor outputs both mean and standard deviation for each action dimension, defining a Gaussian distribution.

2. **Stop Gradients**: The function uses `jax.lax.stop_gradient` to prevent gradient computation through certain paths, ensuring only the actor parameters are differentiated.

3. **Minimum Q-Value**: The minimum of the two critic outputs is used for the policy update to prevent overestimation of Q-values.

4. **Temperature Scaling**: The entropy term is scaled by the temperature parameter, allowing control over the exploration-exploitation trade-off.

5. **Gradient Clipping**: Gradients are clipped to a maximum norm to prevent large parameter updates that could destabilize training.

6. **Return Values**: The function returns:
   - Updated actor parameters
   - Updated optimizer state
   - The actor loss value (scalar)
   - Log probabilities of the sampled actions
   - Standard deviations for the actions

## Verification Tests

To ensure the `actor_update` function works correctly, we've implemented four verification tests that check different aspects of its functionality:

### Basic Functionality Test

The basic functionality test (`test_actor_update_basic`) verifies that:

1. The function executes without errors
2. The optimizer state is properly updated
3. The actor parameters change after the update
4. The returned loss is a scalar value
5. The log probabilities match the expected shape
6. The action standard deviations are valid (positive and in the expected range)

This test also explores how different learning rates affect the actor loss, which helps understand the optimization process.

### Gradient Clipping Test

The gradient clipping test (`test_actor_update_gradient_clipping`) verifies that:

1. Different gradient clipping norms result in different loss values
2. The loss varies predictably with different clipping norms

This test is important because gradient clipping is a key stability mechanism in SAC, especially when dealing with complex policies or environments with high variance rewards.

### Temperature Test

The temperature test (`test_actor_update_temperature`) verifies that:

1. Different temperature values result in different loss values
2. Higher temperature values lead to lower log probabilities (higher entropy)
3. Higher Q-values lead to lower actor loss

This test ensures that the temperature parameter correctly balances the exploration-exploitation trade-off in the actor update, and that the actor properly responds to different Q-value magnitudes.

### Temperature Entropy Test

The temperature entropy test (`test_actor_update_temperature_entropy`) provides a more detailed examination of the relationship between temperature and policy entropy:

1. It tests a wider range of temperature values (from 0.01 to 10)
2. It directly calculates the Gaussian entropy from the actor's action standard deviations
3. It analyzes the correlation between temperature and various policy metrics:
   - Actor loss
   - Log probabilities
   - Action standard deviations
   - Policy entropy

This test is particularly important because policy entropy is a key component of the SAC algorithm that enables:
- Exploration through wider action distributions
- Avoiding premature convergence to suboptimal policies
- Robustness against environment stochasticity

The test visualizes these relationships in comprehensive plots:
- A 2×2 grid showing each metric against temperature
- A comparison plot with normalized metrics to visualize the relative influence of temperature

## Expected Test Results

When the `actor_update` function is implemented correctly, we expect to observe the following:

### Basic Functionality Results
- The actor parameters should change during the update
- The optimizer state should update
- The loss should be a scalar value
- Log probabilities should match the batch shape
- Action standard deviations should be within (0, 1]
- The loss should vary with different learning rates

### Gradient Clipping Results
- Different gradient clipping norms should produce different losses
- Very small clipping norms should substantially restrict parameter updates
- The loss should show a consistent pattern with respect to clipping norms

### Temperature Results
- Higher temperatures should lead to lower log probabilities (higher entropy)
- Higher Q-value scales should produce smaller losses
- The actor should appropriately balance exploration and exploitation

### Temperature Entropy Results
- Higher temperature values should lead to higher policy entropy
- There should be a negative correlation between temperature and log probabilities
- Temperature should have a significant positive correlation with entropy
- The normalized metrics comparison should show clear trade-offs between actor loss, entropy, and log probabilities as temperature changes

## Running the Verification Tests

The verification tests are implemented in `src/agents/functions/verification/actor_update_verification.py`. To run the tests:

```bash
python src/agents/functions/verification/actor_update_verification.py
```

The tests will:
1. Execute all verification checks for the actor_update function
2. Generate visualization plots in the `results/verification/actor_update_verification/` directory
3. Save a CSV file with the test results
4. Display a summary of passing and failing tests

## Conclusion

The `actor_update` function is a crucial component of our SAC implementation, responsible for learning the policy that directly controls agent behavior. Our verification tests confirm that:

1. The function correctly updates the actor parameters to balance Q-value maximization and entropy
2. Gradient clipping works as expected to maintain training stability
3. The temperature parameter properly influences the policy's entropy
4. The actor appropriately responds to different Q-value magnitudes
5. There is a consistent relationship between temperature and policy entropy

By verifying these aspects, we can be confident that our actor updates are mathematically sound and implemented correctly, which is essential for the overall performance and reliability of the SAC algorithm.

The visualization of losses under different learning rates, gradient clipping norms, temperature values, and Q-value scales provides valuable insights into how these hyperparameters affect the learning process, which can guide hyperparameter tuning for specific environments. The detailed temperature-entropy analysis further helps to understand the exploration-exploitation trade-off that is at the core of the SAC algorithm. 