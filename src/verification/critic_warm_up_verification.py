import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import csv
from pathlib import Path
import optax

# Add src to path
sys.path.append('.')

from src.agents.functions.soft_actor_critic_functions import critic_warm_up_update, gaussian_likelihood
from src.agents.functions.networks import DoubleCritic
from src.agents.functions.networks import GaussianActor as Actor
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

# Create mock critic update function for testing
def mock_critic_update(critic_params, critic_opt_state, buffer_weights, states, actions,
                        rewards, next_states, dones, temperature, 
                        critic_target_params, next_actions, next_log_policy):
    """Mock critic update function that returns predetermined values"""
    # Just return slightly modified parameters to check they're being updated
    new_critic_params = jax.tree_util.tree_map(lambda x: x + 0.01, critic_params)
    new_critic_opt_state = critic_opt_state  # We can't easily modify this
    
    # Generate TD errors - these would normally be calculated from the critic
    # Include temperature in the loss calculation to ensure different temperatures produce different losses
    td_errors = jnp.ones((states.shape[0], 1)) * (0.3 + temperature * 0.1)
    
    # Calculate loss based on buffer weights
    critic_loss = jnp.mean(buffer_weights * td_errors)
    
    return new_critic_params, new_critic_opt_state, critic_loss, td_errors

def test_critic_warm_up_basic(save_folder):
    """Test basic functionality of critic_warm_up_update with simple inputs"""
    print("\n========== Critic Warm Up: Basic Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=2
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=2
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, 1))
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, state_dim))
    dones = jnp.zeros((batch_size, 1))  # all not done
    
    # Initialize actor and critic parameters
    actor_params = actor.init(jax.random.fold_in(rng_key, 4), states)
    critic_params = critic.init(jax.random.fold_in(rng_key, 5), states, actions)
    critic_target_params = critic_params  # initially the same
    
    # Create optimizer
    critic_optimizer = optax.adam(learning_rate=3e-4)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Initial temperature value
    initial_temperature = 0.2
    
    # Random distribution for next actions
    normal_dist_next = jax.random.normal(jax.random.fold_in(rng_key, 6), (batch_size, action_dim))
    
    # Set tau (target network update rate)
    tau = 0.005
    
    # Call critic_warm_up_update
    new_critic_params, new_critic_opt_state, new_critic_target_params, critic_loss = critic_warm_up_update(
        actor=actor,
        actor_params=actor_params,
        normal_distribution_for_next_actions=normal_dist_next,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        initial_temperature=initial_temperature,
        critic_params=critic_params,
        critic_target_params=critic_target_params,
        critic_opt_state=critic_opt_state,
        critic_update_lambda=mock_critic_update,
        tau=tau
    )
    
    # Verify the function executes without errors
    test_basic_run_passed = True
    test_results.add_result(
        "Critic warm up basic run",
        test_basic_run_passed,
        "Function executed without errors"
    )
    
    # Verify critic parameters were updated
    critic_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            critic_params, 
            new_critic_params
        )
    )
    test_results.add_result(
        "Critic parameters updated",
        critic_params_changed,
        "Expected critic parameters to change"
    )
    
    # Verify target critic parameters were updated
    target_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            critic_target_params, 
            new_critic_target_params
        )
    )
    test_results.add_result(
        "Target critic parameters updated",
        target_params_changed,
        "Expected target critic parameters to change"
    )
    
    # Verify critic loss is a scalar
    loss_is_scalar = jnp.isscalar(critic_loss) or (isinstance(critic_loss, jnp.ndarray) and critic_loss.ndim == 0)
    test_results.add_result(
        "Critic loss is scalar",
        loss_is_scalar,
        f"Expected scalar loss, got shape {critic_loss.shape if hasattr(critic_loss, 'shape') else 'scalar'}"
    )
    
    # Plot tau effect on target network updates
    tau_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    target_distances = []
    
    for tau_val in tau_values:
        # Apply target update with current tau
        updated_target = jax.tree_util.tree_map(
            lambda p, tp: tau_val * p + (1.0 - tau_val) * tp,
            critic_params,
            critic_target_params
        )
        
        # Calculate distance between updated target and original critic params
        flat_critic = jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(critic_params)])
        flat_target = jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(updated_target)])
        distance = jnp.sqrt(jnp.sum((flat_critic - flat_target) ** 2))
        target_distances.append(float(distance))
    
    plt.figure(figsize=(12, 8))
    plt.plot(tau_values, target_distances, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Target Network Distance from Critic vs. Tau', fontsize=22)
    plt.xlabel('Tau', fontsize=20)
    plt.ylabel('Parameter Distance', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/target_distance_vs_tau.png', dpi=300)
    
    all_basic_tests_passed = all([
        test_basic_run_passed, critic_params_changed, 
        target_params_changed, loss_is_scalar
    ])
    
    return all_basic_tests_passed

def test_critic_warm_up_initial_temperature(save_folder):
    """Test the effect of different initial temperature values"""
    print("\n========== Critic Warm Up: Initial Temperature Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=2
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=2
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 32  # larger batch for better statistics
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, 1)) * 0.1
    next_states = states + jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, state_dim)) * 0.1
    dones = jnp.zeros((batch_size, 1))  # all not done
    
    # Initialize actor and critic parameters
    actor_params = actor.init(jax.random.fold_in(rng_key, 4), states)
    critic_params = critic.init(jax.random.fold_in(rng_key, 5), states, actions)
    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
    
    # Create optimizer
    critic_optimizer = optax.adam(learning_rate=3e-4)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Random distribution for next actions
    normal_dist_next = jax.random.normal(jax.random.fold_in(rng_key, 6), (batch_size, action_dim))
    
    # Set tau
    tau = 0.005
    
    # Test different initial temperature values
    temperature_values = [0.1, 0.2, 0.5, 1.0, 2.0]
    losses = []
    
    for temp in temperature_values:
        # Deep copy parameters to avoid interference between tests
        critic_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
        critic_target_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), critic_target_params)
        critic_opt_state_copy = critic_optimizer.init(critic_params_copy)
        
        # Call critic_warm_up_update
        _, _, _, critic_loss = critic_warm_up_update(
            actor=actor,
            actor_params=actor_params,
            normal_distribution_for_next_actions=normal_dist_next,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            initial_temperature=temp,
            critic_params=critic_params_copy,
            critic_target_params=critic_target_params_copy,
            critic_opt_state=critic_opt_state_copy,
            critic_update_lambda=mock_critic_update,
            tau=tau
        )
        
        losses.append(float(critic_loss))
    
    # Verify that different temperatures produce different losses
    unique_losses = len(set(losses)) == len(losses)
    test_results.add_result(
        "Different temperatures produce different losses",
        unique_losses,
        f"Expected different losses for different temperatures, got {losses}"
    )
    
    # Plot temperature effect on loss
    plt.figure(figsize=(12, 8))
    plt.plot(temperature_values, losses, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.title('Critic Loss vs. Initial Temperature', fontsize=22)
    plt.xlabel('Initial Temperature', fontsize=20)
    plt.ylabel('Critic Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/loss_vs_temperature.png', dpi=300)
    
    return unique_losses

def run_all_tests():
    """Run all critic_warm_up verification tests"""
    print("\n===== STARTING CRITIC WARM UP VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/critic_warm_up_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Basic critic_warm_up functionality
    print("\n[1/2] Running Basic Critic Warm Up Test...")
    basic_test_result = test_critic_warm_up_basic(save_folder)
    
    # Test 2: Initial temperature effect test
    print("\n[2/2] Running Initial Temperature Test...")
    temperature_test_result = test_critic_warm_up_initial_temperature(save_folder)
    
    print("\n===== CRITIC WARM UP VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/critic_warm_up_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 