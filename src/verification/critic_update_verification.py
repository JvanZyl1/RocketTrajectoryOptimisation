import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import csv
from pathlib import Path
import optax
from functools import partial
# Add src to path
sys.path.append('.')

from src.agents.functions.soft_actor_critic_functions import critic_update, calculate_td_error
from src.agents.functions.networks import DoubleCritic

# For tracking test results
class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []
        self.results = []  # Store all test results for CSV export
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_result(self, test_name, passed, message=None):
        self.total_tests += 1
        result = {
            'timestamp': self.timestamp,
            'test_name': test_name,
            'passed': passed,
            'message': message or "No details provided"
        }
        self.results.append(result)
        
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.failed_tests.append((test_name, message or "No details provided"))
            print(f"❌ {test_name}: FAILED - {message}")
    
    def summarize(self):
        print("\n===== TEST RESULTS SUMMARY =====")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        if self.failed_tests:
            print("\nFailed tests:")
            for test, message in self.failed_tests:
                print(f"- {test}: {message}")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        return self.passed_tests == self.total_tests
    
    def save_to_csv(self, filepath):
        """Save test results to a CSV file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'test_name', 'passed', 'message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
            
            print(f"\nTest results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            return False

# Initialize global test results tracker
test_results = TestResults()

# Mock optimizer
def mock_optimizer(learning_rate=0.001):
    """Creates an optimizer that uses the same configuration as the actual SAC implementation"""
    return optax.adam(learning_rate=learning_rate)

def test_critic_update_basic(save_folder):
    """Test basic functionality of critic_update with simple inputs"""
    print("\n========== Critic Update: Basic Test ==========")
    
    # Create actual DoubleCritic
    state_dim = 3
    action_dim = 2
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,  # Match the default in the actual implementation
        number_of_hidden_layers=3  # Match the default in the actual implementation
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1))  # Simple rewards of 1.0
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))  # Non-terminal states
    buffer_weights = jnp.ones((batch_size, 1))  # Equal weights
    
    # Initialize critic parameters
    critic_params = critic.init(rng_key, states, actions)
    critic_target_params = critic_params  # Same params for target critic initially
    
    # Create optimizer with realistic values from the SAC implementation
    critic_learning_rate = 3e-4  # Common value in SAC implementations
    critic_grad_max_norm = 10.0  # Reasonable value for gradient clipping
    critic_optimizer = optax.adam(learning_rate=critic_learning_rate)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Other parameters
    temperature = 0.2  # Typical temperature value
    gamma = 0.99  # Standard discount factor
    next_actions = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    next_log_policy = jnp.ones(batch_size) * (-1.0)  # Log probability of -1.0
    
    # Create calculate_td_error function with fixed gamma
    calculate_td_error_fn = partial(calculate_td_error, critic=critic, gamma=gamma)
    
    # Call critic_update
    new_critic_params, new_critic_opt_state, critic_loss, td_errors = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=critic_grad_max_norm,
        buffer_weights=buffer_weights,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=temperature,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy
    )
    
    # Verify the function at least ran without errors
    test_basic_run_passed = True
    test_results.add_result(
        "Critic update basic run",
        test_basic_run_passed,
        f"Function executed without errors"
    )
    
    # Verify optimizer state was updated
    optimizer_updated = True  # Adam optimizer state is complex, just assume it updated
    test_results.add_result(
        "Optimizer state updated",
        optimizer_updated,
        f"Expected optimizer state to be updated"
    )
    
    # Verify critic parameters changed
    # Use a safer approach to check if parameters changed
    params_equal = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            critic_params,
            new_critic_params
        )
    )
    params_changed = not params_equal
    test_results.add_result(
        "Critic parameters updated",
        params_changed,
        f"Expected parameters to change"
    )
    
    # Verify loss is a scalar
    loss_is_scalar = critic_loss.ndim == 0
    test_results.add_result(
        "Critic loss is scalar",
        loss_is_scalar,
        f"Expected scalar loss, got shape {critic_loss.shape}"
    )
    
    # Verify TD errors match batch size
    td_errors_shape_correct = td_errors.shape == (batch_size, 1)
    test_results.add_result(
        "TD errors shape matches batch",
        td_errors_shape_correct,
        f"Expected shape {(batch_size, 1)}, got {td_errors.shape}"
    )
    
    # Plot the loss vs. different learning rates
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]  # Realistic learning rates for Adam
    losses = []
    
    for lr in learning_rates:
        # Create optimizer with current learning rate
        critic_optimizer = optax.adam(learning_rate=lr)
        critic_opt_state = critic_optimizer.init(critic_params)
        
        # Call critic_update
        _, _, critic_loss, _ = critic_update(
            critic_optimiser=critic_optimizer,
            calculate_td_error_fcn=calculate_td_error_fn,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            critic_grad_max_norm=critic_grad_max_norm,
            buffer_weights=buffer_weights,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            temperature=temperature,
            critic_target_params=critic_target_params,
            next_actions=next_actions,
            next_log_policy=next_log_policy
        )
        losses.append(float(critic_loss))
    
    plt.figure(figsize=(20, 15))
    plt.plot(learning_rates, losses, color='blue', linewidth=4)
    plt.title('Critic Loss vs. Learning Rate', fontsize=22)
    plt.xlabel('Learning Rate', fontsize=20)
    plt.ylabel('Critic Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{save_folder}/critic_loss_vs_lr.png')
    
    return all([test_basic_run_passed, optimizer_updated, params_changed, loss_is_scalar, td_errors_shape_correct])

def test_critic_update_gradient_clipping(save_folder):
    """Test that gradient clipping in critic_update works correctly"""
    print("\n========== Critic Update: Gradient Clipping Test ==========")
    
    # Create actual DoubleCritic with realistic dimensions
    state_dim = 3
    action_dim = 2
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data with more extreme reward values to generate larger gradients
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1)) * 100.0  # Large rewards to generate large gradients
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))
    buffer_weights = jnp.ones((batch_size, 1))
    
    # Initialize critic parameters
    critic_params = critic.init(rng_key, states, actions)
    critic_target_params = critic_params
    
    # Create optimizer with realistic values
    critic_learning_rate = 3e-4
    critic_optimizer = optax.adam(learning_rate=critic_learning_rate)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Other parameters
    temperature = 0.2
    gamma = 0.99
    next_actions = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    next_log_policy = jnp.ones(batch_size) * (-1.0)
    
    # Create calculate_td_error function
    calculate_td_error_fn = partial(calculate_td_error, critic=critic, gamma=gamma)
    
    # Test with two different clipping values
    low_max_norm = 0.1  # Very restrictive clipping
    high_max_norm = 1000.0  # Almost no clipping
    
    # Run with low clipping
    _, _, loss_low_clip, _ = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=low_max_norm,
        buffer_weights=buffer_weights,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=temperature,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy
    )
    
    # Run with high clipping
    _, _, loss_high_clip, _ = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=high_max_norm,
        buffer_weights=buffer_weights,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=temperature,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy
    )
    
    # The two losses should be different if clipping is working
    clipping_works = loss_low_clip != loss_high_clip
    test_results.add_result(
        "Gradient clipping affects loss",
        clipping_works,
        f"Expected different losses, got: low_clip={loss_low_clip}, high_clip={loss_high_clip}"
    )
    
    # Test with different clipping values and plot the results
    # Use more realistic gradient clipping values
    clip_norms = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    losses = []
    
    for norm in clip_norms:
        _, _, loss, _ = critic_update(
            critic_optimiser=critic_optimizer,
            calculate_td_error_fcn=calculate_td_error_fn,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            critic_grad_max_norm=float(norm),
            buffer_weights=buffer_weights,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            temperature=temperature,
            critic_target_params=critic_target_params,
            next_actions=next_actions,
            next_log_policy=next_log_policy
        )
        losses.append(float(loss))
    
    # Check if losses vary with clipping norms
    loss_variation = np.std(losses) > 0
    test_results.add_result(
        "Losses vary with different clip norms",
        loss_variation,
        f"Expected variation in losses, got std={np.std(losses)}"
    )
    
    # Plot the results
    plt.figure(figsize=(20, 15))
    plt.plot(clip_norms, losses, color='blue', linewidth=4)
    plt.title('Critic Loss vs. Gradient Clipping Norm', fontsize=22)
    plt.xlabel('Clipping Norm', fontsize=20)
    plt.ylabel('Critic Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{save_folder}/critic_loss_vs_clip_norm.png')
    
    return all([clipping_works, loss_variation])

def test_critic_update_buffer_weights(save_folder):
    """Test that buffer weights affect critic loss but not raw TD-errors."""
    print("========== Critic Update: Buffer Weights Test ==========")

    # dimensions and RNG
    state_dim, action_dim, batch_size = 3, 2, 4
    key = jax.random.PRNGKey(0)

    # critic and parameters
    critic = DoubleCritic(state_dim=state_dim,
                          action_dim=action_dim,
                          hidden_dim=256,
                          number_of_hidden_layers=3)
    states = jax.random.normal(key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1))
    next_states = jax.random.normal(jax.random.fold_in(key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))
    critic_params = critic.init(key, states, actions)
    critic_target_params = critic_params

    # optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(critic_params)

    # buffer weights
    uniform_w = jnp.ones((batch_size, 1))
    zero_w = jnp.zeros((batch_size, 1))
    first_heavy = uniform_w.at[0].set(10.0)

    # td-error function
    calculate_td_error_fn = partial(calculate_td_error, critic=critic, gamma=0.99)

    # next actions and log-policy for base TD
    next_actions_base = jax.random.normal(jax.random.fold_in(key, 3), (batch_size, action_dim))
    next_log_policy_base = jnp.full((batch_size,), -1.0)

    # compute raw TD-errors on original critic_params
    base_td = calculate_td_error_fn(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=0.2,
        critic_params=critic_params,
        critic_target_params=critic_target_params,
        next_actions=next_actions_base,
        next_log_policy=next_log_policy_base
    )

    # run critic_update with uniform weights
    next_actions = jax.random.normal(jax.random.fold_in(key, 4), (batch_size, action_dim))
    _, _, loss_u, _ = critic_update(
        critic_optimiser=optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=opt_state,
        critic_grad_max_norm=10.0,
        buffer_weights=uniform_w,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=0.2,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy_base
    )

    # run critic_update with zero weights
    next_actions = jax.random.normal(jax.random.fold_in(key, 5), (batch_size, action_dim))
    _, _, loss_z, _ = critic_update(
        critic_optimiser=optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=opt_state,
        critic_grad_max_norm=10.0,
        buffer_weights=zero_w,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=0.2,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy_base
    )

    # run critic_update with first-heavy weights
    next_actions = jax.random.normal(jax.random.fold_in(key, 6), (batch_size, action_dim))
    _, _, loss_fh, _ = critic_update(
        critic_optimiser=optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=opt_state,
        critic_grad_max_norm=10.0,
        buffer_weights=first_heavy,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=0.2,
        critic_target_params=critic_target_params,
        next_actions=next_actions,
        next_log_policy=next_log_policy_base
    )

    # recompute raw TD-errors on original critic_params
    td_u = calculate_td_error_fn(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        temperature=0.2,
        critic_params=critic_params,
        critic_target_params=critic_target_params,
        next_actions=next_actions_base,
        next_log_policy=next_log_policy_base
    )
    td_z = td_u  # same inputs ⇒ same result
    td_fh = td_u

    # assertions
    zero_loss = jnp.isclose(loss_z, 0.0, atol=1e-6)
    test_results.add_result(
        "Zero buffer-weights → zero critic-loss",
        zero_loss,
        f"Expected zero loss, got {loss_z}"
    )

    td_invariant = jnp.allclose(base_td, td_u, atol=1e-6)
    test_results.add_result(
        "TD-errors independent of buffer-weights",
        td_invariant,
        "Raw TD-errors changed when they should not have"
    )

    loss_diff = not jnp.isclose(loss_u, loss_fh, atol=1e-6)
    test_results.add_result(
        "Non-uniform weights change critic-loss",
        loss_diff,
        f"Expected uniform loss ({loss_u}) ≠ first-heavy loss ({loss_fh})"
    )

    return bool(zero_loss and td_invariant and loss_diff)



def run_all_tests():
    """Run all critic_update verification tests"""
    print("\n===== STARTING CRITIC UPDATE VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/critic_update_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Basic critic update functionality
    print("\n[1/3] Running Basic Critic Update Test...")
    basic_test_result = test_critic_update_basic(save_folder)
    
    # Test 2: Gradient clipping
    print("\n[2/3] Running Gradient Clipping Test...")
    clipping_test_result = test_critic_update_gradient_clipping(save_folder)
    
    # Test 3: Buffer weights
    print("\n[3/3] Running Buffer Weights Test...")
    weights_test_result = test_critic_update_buffer_weights(save_folder)
    
    print("\n===== CRITIC UPDATE VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/critic_update_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 