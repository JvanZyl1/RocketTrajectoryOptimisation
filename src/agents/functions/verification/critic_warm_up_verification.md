# Critic Warm Up Verification

## Purpose
The `critic_warm_up_update` function is a crucial component of the Soft Actor-Critic (SAC) algorithm that handles the initial training phase of the critic network. This function:
1. Samples next actions using the actor network
2. Computes log probabilities for these actions
3. Updates the critic network using Temporal Difference (TD) learning
4. Updates the target critic network using a soft update rule

## Mathematical Foundation

### Target Network Update
The target network is updated using a soft update rule:
```
target_params = τ * critic_params + (1 - τ) * target_params
```
where τ (tau) is the target network update rate, typically a small value (e.g., 0.005).

### Temperature Effect
The initial temperature parameter affects the exploration-exploitation trade-off:
- Lower temperature values lead to more deterministic actions
- Higher temperature values encourage more exploration
- The temperature affects the TD error calculation through the entropy term
- The critic loss is proportional to the temperature value: `loss = base_loss + temperature * scale_factor`

## Key Aspects for Verification

1. **Basic Functionality**
   - The function should execute without errors
   - Critic parameters should be updated
   - Target network parameters should be updated
   - Loss should be a scalar value

2. **Target Network Updates**
   - The distance between critic and target networks should increase with τ
   - Updates should be smooth and continuous
   - The target network should track the critic network with a delay

3. **Temperature Effects**
   - Different temperature values should produce different losses
   - The relationship between temperature and loss should be linear
   - The loss should increase with higher temperature values
   - The base loss should be non-zero even at zero temperature

## Test Structure

The verification tests are implemented in `critic_warm_up_verification.py` and include:

1. **Basic Test** (`test_critic_warm_up_basic`)
   - Tests basic functionality with simple inputs
   - Verifies parameter updates
   - Checks loss calculation
   - Generates visualization of target network distance vs. τ

2. **Initial Temperature Test** (`test_critic_warm_up_initial_temperature`)
   - Tests the effect of different temperature values
   - Verifies that different temperatures produce different losses
   - Ensures the loss increases with temperature
   - Generates visualization of loss vs. temperature

## Running the Tests

To run the verification tests:

```bash
python src/agents/functions/verification/critic_warm_up_verification.py
```

The tests will:
1. Execute all verification tests
2. Generate visualizations in the `results/verification/critic_warm_up_verification` directory
3. Save test results to a timestamped CSV file
4. Display a summary of test results

## Test Output

The tests generate:
1. A summary of passed/failed tests in the console
2. A CSV file with detailed test results
3. Two visualization files:
   - `target_distance_vs_tau.png`: Shows how target network distance varies with τ
   - `loss_vs_temperature.png`: Shows how critic loss varies with temperature

## Expected Results

All tests should pass, indicating that:
1. The critic warm up process executes correctly
2. Parameter updates are applied properly
3. Target network updates follow the expected pattern
4. Temperature affects the learning process as expected:
   - Higher temperatures result in higher critic losses
   - The relationship between temperature and loss is monotonic
   - The base loss is maintained even at zero temperature 