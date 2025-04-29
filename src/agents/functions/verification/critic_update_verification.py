import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import csv
from pathlib import Path
import flax.linen as nn
import optax
from functools import partial
from typing import Callable, Tuple

# Add src to path
sys.path.append('.')

from src.agents.functions.soft_actor_critic_functions import critic_update, calculate_td_error, clip_grads

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

# Simple mock critic for testing
class MockDoubleCritic(nn.Module):
    @nn.compact
    def __call__(self, states, actions):
        # Return predetermined Q-values for testing
        batch_size = states.shape[0]
        q1 = jnp.ones((batch_size, 1)) * 5.0  # Fixed Q1 value of 5.0
        q2 = jnp.ones((batch_size, 1)) * 4.0  # Fixed Q2 value of 4.0
        return q1, q2

# Create mock optimizer
def mock_optimizer(learning_rate=0.001):
    """Creates an optimizer for testing that updates in a predictable way"""
    def init_fn(params):
        return {'learning_rate': learning_rate, 'count': 0}
    
    def update_fn(updates, state, params=None):
        new_state = {'learning_rate': state['learning_rate'], 'count': state['count'] + 1}
        # Scale updates by learning rate for predictable changes
        scaled_updates = jax.tree_util.tree_map(lambda g: -state['learning_rate'] * g, updates)
        return scaled_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)

def test_critic_update_basic(save_folder):
    """Test basic functionality of critic_update with simple inputs"""
    print("\n========== Critic Update: Basic Test ==========")
    
    # Create mock double critic
    critic = MockDoubleCritic()
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    state_dim = 3
    action_dim = 2
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1))  # Simple rewards of 1.0
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))  # Non-terminal states
    buffer_weights = jnp.ones((batch_size, 1))  # Equal weights
    
    # Mock critic parameters
    critic_params = critic.init(rng_key, states, actions)
    critic_target_params = critic_params  # Same params for target critic initially
    
    # Create optimizer
    test_lr = 0.01
    critic_optimizer = mock_optimizer(learning_rate=test_lr)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Other parameters
    temperature = 0.1
    critic_grad_max_norm = 1.0
    gamma = 0.99
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
    optimizer_updated = new_critic_opt_state['count'] > critic_opt_state['count']
    test_results.add_result(
        "Optimizer state updated",
        optimizer_updated,
        f"Expected update count to increase, got {new_critic_opt_state['count']}"
    )
    
    # Verify critic parameters changed
    # Compare flatten parameters
    old_flat_params = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(critic_params)])
    new_flat_params = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(new_critic_params)])
    params_changed = not jnp.allclose(old_flat_params, new_flat_params)
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
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    losses = []
    
    for lr in learning_rates:
        # Create optimizer with current learning rate
        critic_optimizer = mock_optimizer(learning_rate=lr)
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
    
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, losses, 'o-', linewidth=2)
    plt.title('Critic Loss vs. Learning Rate', fontsize=18)
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Critic Loss', fontsize=14)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f'{save_folder}/critic_loss_vs_lr.png')
    
    return all([test_basic_run_passed, optimizer_updated, params_changed, loss_is_scalar, td_errors_shape_correct])

def test_critic_update_gradient_clipping(save_folder):
    """Test that gradient clipping in critic_update works correctly"""
    print("\n========== Critic Update: Gradient Clipping Test ==========")
    
    # Create mock critic
    critic = MockDoubleCritic()
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 2
    state_dim = 3
    action_dim = 2
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1)) * 100.0  # Large rewards to generate large gradients
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))
    buffer_weights = jnp.ones((batch_size, 1))
    
    # Mock critic parameters
    critic_params = critic.init(rng_key, states, actions)
    critic_target_params = critic_params
    
    # Create optimizer
    test_lr = 0.01
    critic_optimizer = mock_optimizer(learning_rate=test_lr)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Other parameters
    temperature = 0.1
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
    clip_norms = np.logspace(-3, 3, 10)  # Log scale from 0.001 to 1000
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
    plt.figure(figsize=(12, 8))
    plt.plot(clip_norms, losses, 'o-', linewidth=2)
    plt.title('Critic Loss vs. Gradient Clipping Norm', fontsize=18)
    plt.xlabel('Clipping Norm', fontsize=14)
    plt.ylabel('Critic Loss', fontsize=14)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f'{save_folder}/critic_loss_vs_clip_norm.png')
    
    return all([clipping_works, loss_variation])

def test_critic_update_buffer_weights(save_folder):
    """Test that buffer weights properly affect the update in critic_update"""
    print("\n========== Critic Update: Buffer Weights Test ==========")
    
    # Create mock critic
    critic = MockDoubleCritic()
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    state_dim = 3
    action_dim = 2
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jnp.ones((batch_size, 1))
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))
    
    # Create different buffer weight scenarios
    uniform_weights = jnp.ones((batch_size, 1))  # Equal weights
    zero_weights = jnp.zeros((batch_size, 1))  # Zero weights (should give zero loss)
    first_heavy = jnp.array([[10.0], [1.0], [1.0], [1.0]])  # First sample weighted 10x
    
    # Mock critic parameters
    critic_params = critic.init(rng_key, states, actions)
    critic_target_params = critic_params
    
    # Create optimizer
    test_lr = 0.01
    critic_optimizer = mock_optimizer(learning_rate=test_lr)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Other parameters
    temperature = 0.1
    critic_grad_max_norm = 1.0
    gamma = 0.99
    next_actions = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    next_log_policy = jnp.ones(batch_size) * (-1.0)
    
    # Create calculate_td_error function
    calculate_td_error_fn = partial(calculate_td_error, critic=critic, gamma=gamma)
    
    # Run with uniform weights
    _, _, loss_uniform, td_errors_uniform = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=critic_grad_max_norm,
        buffer_weights=uniform_weights,
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
    
    # Run with zero weights
    _, _, loss_zero, td_errors_zero = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=critic_grad_max_norm,
        buffer_weights=zero_weights,
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
    
    # Run with first sample weighted 10x
    _, _, loss_first_heavy, td_errors_first_heavy = critic_update(
        critic_optimiser=critic_optimizer,
        calculate_td_error_fcn=calculate_td_error_fn,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        critic_grad_max_norm=critic_grad_max_norm,
        buffer_weights=first_heavy,
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
    
    # Zero weights should give zero loss
    zero_weights_give_zero_loss = jnp.isclose(loss_zero, 0.0, atol=1e-6)
    test_results.add_result(
        "Zero weights give zero loss",
        zero_weights_give_zero_loss,
        f"Expected loss close to 0.0, got {loss_zero}"
    )
    
    # TD errors should be the same regardless of weights
    td_errors_independent = (jnp.allclose(td_errors_uniform, td_errors_zero, atol=1e-6) and 
                             jnp.allclose(td_errors_uniform, td_errors_first_heavy, atol=1e-6))
    test_results.add_result(
        "TD errors independent of weights",
        td_errors_independent,
        f"Expected TD errors to be independent of weights"
    )
    
    # First heavy weights should give different loss than uniform
    first_heavy_different = not jnp.isclose(loss_uniform, loss_first_heavy, atol=1e-6)
    test_results.add_result(
        "Non-uniform weights affect loss",
        first_heavy_different,
        f"Expected different losses: uniform={loss_uniform}, first_heavy={loss_first_heavy}"
    )
    
    # Plot the effect of different weight distributions
    weight_factors = np.linspace(0, 20, 20)  # First weight from 0x to 20x
    losses = []
    
    for factor in weight_factors:
        weights = jnp.ones((batch_size, 1))
        weights = weights.at[0].set(factor)
        
        _, _, loss, _ = critic_update(
            critic_optimiser=critic_optimizer,
            calculate_td_error_fcn=calculate_td_error_fn,
            critic_params=critic_params,
            critic_opt_state=critic_opt_state,
            critic_grad_max_norm=critic_grad_max_norm,
            buffer_weights=weights,
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
    
    plt.figure(figsize=(12, 8))
    plt.plot(weight_factors, losses, 'o-', linewidth=2)
    plt.title('Critic Loss vs. First Sample Weight Factor', fontsize=18)
    plt.xlabel('First Sample Weight Factor', fontsize=14)
    plt.ylabel('Critic Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(f'{save_folder}/critic_loss_vs_weight.png')
    
    return all([zero_weights_give_zero_loss, td_errors_independent, first_heavy_different])

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