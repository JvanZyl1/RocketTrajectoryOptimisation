# Temperature Update Verification

The `temperature_update` function in the SAC (Soft Actor-Critic) algorithm is responsible for dynamically adjusting the temperature parameter (α) to maintain policy entropy at a target level.

## Mathematical Foundation

The goal of the temperature update is to minimize the following loss function:

$L(α) = -α * (log\_probs + target\_entropy)$

where:
- α (alpha) is the temperature parameter
- log_probs are the log probabilities of the actions sampled from the policy
- target_entropy is usually set as the negative action dimension (-dim(A))

In the optimization process:
- The temperature parameter is represented internally as log(α) to ensure α remains positive
- An optimizer (typically Adam) updates log(α) based on the gradient of the loss
- The learning rate of the optimizer controls how quickly the temperature changes

## Expected Behavior

Our empirical observations show that:

1. **Temperature Convergence**: The temperature generally tends to decrease over time during training regardless of initial conditions, but at different rates.

2. **Loss Variation**:
   - **Target Entropy**: More negative target entropy values result in more negative loss values. We observe a positive correlation between target entropy and loss.
   - **Log Probabilities**: More negative log probabilities also result in more negative loss values. There is a positive correlation between log probabilities and loss.

3. **Temperature Changes**: 
   - When log_probs + target_entropy = 0 (equilibrium point), the temperature should change at a slower rate compared to cases where this sum is far from zero.

## Verification Tests

The verification tests check these behaviors:

1. **Basic Test**: Ensures that the temperature update function runs without errors, updates the optimizer state, computes a loss, and returns valid values.

2. **Target Entropy Test**: Verifies that different target entropy values produce different loss values and rates of change in temperature. We expect to see a positive correlation between target entropy and loss values, as more negative target entropy values lead to more negative loss values.

3. **Log Probabilities Test**: Checks that different log probability values produce different loss values and rates of change in temperature. We expect to see a positive correlation between log probabilities and loss, as more negative log probabilities lead to more negative loss values.

## Visualization

The tests generate several visualizations:

1. Temperature vs. Target Entropy/Log Probability plots showing the final temperature after multiple updates with different parameters.

2. Loss vs. Target Entropy/Log Probability plots showing how the loss function changes with different parameters.

3. Temperature Change Over Time plots showing how the temperature evolves during multiple update steps for different parameters.

These visualizations help us understand the dynamics of the temperature update process and verify that the implementation behaves as expected. 