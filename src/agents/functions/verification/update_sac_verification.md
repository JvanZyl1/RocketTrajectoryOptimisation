# Update SAC Verification

The `update_sac` function is the central orchestration function in the Soft Actor-Critic (SAC) algorithm that coordinates the updates of the critic, actor, and temperature parameters. This document outlines the verification tests designed to ensure it functions correctly.

## Function Purpose

In the SAC algorithm, the `update_sac` function has several key responsibilities:

1. **Sampling Next Actions**: Using the current actor policy to generate actions for the next states
2. **Computing Log Probabilities**: Calculating log probabilities for the sampled next actions
3. **Critic Update**: Updating the critic networks using TD targets and computing TD errors
4. **Actor Update**: Updating the actor network to maximize expected returns and entropy
5. **Temperature Update**: Adjusting the temperature parameter to maintain a target entropy level
6. **Target Network Update**: Soft-updating the target critic network using a weighted average
7. **Integration**: Ensuring all individual components work together correctly

## Mathematical Foundation

The SAC algorithm involves the following update equations:

1. **Critic Update**: Minimizes TD error between Q-values and bootstrapped targets
   ```
   TD target = r + γ(1-d)(min(Q₁',Q₂') - α*log π(a'|s'))
   Loss = E[w*(TD target - Q(s,a))²]
   ```
   where w represents the buffer weights for prioritized experience replay.

2. **Actor Update**: Maximizes expected Q-value and entropy
   ```
   J(φ) = E[min(Q₁,Q₂) - α*log π(a|s)]
   ```

3. **Temperature Update**: Adjusts temperature to maintain target entropy
   ```
   J(α) = -α*(log π(a|s) + target_entropy)
   ```

4. **Target Update**: Soft update of target critic parameters
   ```
   θ' = τ*θ + (1-τ)*θ'
   ```

## Key Aspects for Verification

The verification tests focus on the following critical aspects:

1. **Basic Functionality**: Ensuring the function runs without errors and returns the expected outputs
2. **Parameter Updates**: Verifying that critic, actor, target critic, and temperature parameters are correctly updated
3. **Loss Calculations**: Checking that critic, actor, and temperature losses are computed correctly
4. **First-Step Behavior**: Testing that temperature updates are skipped during the first optimization step
5. **Buffer Weights**: Verifying that prioritized experience replay weights are properly applied to the critic update
6. **Target Network Updates**: Ensuring the target network is updated with the correct interpolation factor (tau)
7. **Integration Testing**: Confirming that all components work together correctly

## Verification Tests

### Basic Test (`test_update_sac_basic`)

This test checks the fundamental functionality of the `update_sac` function:

1. The function executes without errors with realistic inputs
2. Critic parameters are correctly updated
3. Actor parameters are correctly updated
4. Target critic parameters are updated through soft updates
5. Temperature is unchanged during the first update step (when `first_step_bool=True`)
6. Temperature loss is zero during the first update step
7. Temperature is properly updated in subsequent steps
8. Temperature loss is non-zero in subsequent steps
9. The effect of different tau values on target network updates is visualized

### Integration Test (`test_update_sac_integration`)

This test verifies how different components interact in the `update_sac` function:

1. Tests with different buffer weight configurations:
   - Uniform weights (all weights equal to 1)
   - Prioritized weights (varying importance weights from 0.1 to 2.0)
   - Zero weights (all weights equal to 0)
   
2. Verifies that:
   - Zero weights result in zero critic loss (confirmed)
   - Different weight schemes produce different critic losses (confirmed)
   - Actor loss is consistent across weight schemes (confirmed)
   - Temperature loss is consistent across weight schemes (confirmed)

3. Metrics are visualized to compare performance across different buffer weight schemes

## Test Results

The verification tests have been successfully completed with the following results:

1. **Basic Functionality**: All basic tests passed, confirming proper parameter updates and first-step behavior
2. **Integration Tests**: All integration tests passed, demonstrating correct handling of buffer weights
3. **Weight Effects**: Tests confirmed that:
   - Zero weights correctly result in zero critic loss
   - Prioritized weights (0.1 to 2.0) produce different critic losses than uniform weights
   - Actor and temperature updates remain consistent across weight schemes

## Visual Analysis

The verification tests include visualizations to help understand the behavior of the `update_sac` function:

1. **Target Distance vs. Tau**: Shows how different tau values affect the distance between critic and target critic parameters
2. **Metrics Comparison Across Weight Schemes**: Compares critic loss, actor loss, temperature loss, TD errors, and log probabilities across different buffer weight configurations

## Expected Behavior

When functioning correctly, the `update_sac` function should:

1. Successfully update all parameters (critic, actor, temperature, target critic)
2. Skip temperature updates during the first step
3. Apply proper weighting of TD errors in critic loss calculation
4. Maintain the independence of actor and temperature updates from buffer weights
5. Apply soft updates to target network parameters
6. Coordinate the interaction between critic, actor, and temperature updates
7. Return all expected outputs with appropriate shapes and values

## Running the Verification Tests

The verification tests are implemented in `src/agents/functions/verification/update_sac_verification.py`. To run these tests:

```bash
python src/agents/functions/verification/update_sac_verification.py
```

The tests will:
1. Generate visualizations in the `results/verification/update_sac_verification/` directory
2. Save test results to a CSV file with timestamp
3. Display a summary of passing and failing tests
4. Return a non-zero exit code if any tests fail

## Test Output Format

The test results include:
- Detailed metrics for each weight configuration
- Visual plots of parameter updates and weight effects
- A summary of all test outcomes
- CSV file with timestamped results for tracking changes over time 