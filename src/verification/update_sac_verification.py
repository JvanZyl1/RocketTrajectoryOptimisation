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

from src.agents.functions.soft_actor_critic_functions import update_sac, gaussian_likelihood
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

# Create mock update functions to test the integration
def mock_critic_update(critic_params, critic_opt_state, buffer_weights, states, actions,
                        rewards, next_states, dones, temperature, 
                        critic_target_params, next_actions, next_log_policy):
    """Mock critic update function that returns predetermined values"""
    # Just return slightly modified parameters to check they're being updated
    new_critic_params = jax.tree_util.tree_map(lambda x: x + 0.01, critic_params)
    new_critic_opt_state = critic_opt_state  # We can't easily modify this
    
    # Generate TD errors - these would normally be calculated from the critic
    td_errors = jnp.ones((states.shape[0], 1)) * 0.3
    
    # Apply buffer weights to calculate the loss
    # This properly simulates how weights affect the critic loss
    critic_loss = jnp.mean(buffer_weights * td_errors)
    
    return new_critic_params, new_critic_opt_state, critic_loss, td_errors

def mock_actor_update(temperature, states, normal_distribution, critic_params, 
                      actor_params, actor_opt_state):
    """Mock actor update function that returns predetermined values"""
    # Just return slightly modified parameters to check they're being updated
    new_actor_params = jax.tree_util.tree_map(lambda x: x + 0.01, actor_params)
    new_actor_opt_state = actor_opt_state  # We can't easily modify this
    actor_loss = jnp.array(0.3)
    log_probs = jnp.ones((states.shape[0],)) * -1.0
    action_std = jnp.ones((states.shape[0], 2)) * 0.5
    return new_actor_params, new_actor_opt_state, actor_loss, log_probs, action_std

def mock_temperature_update(current_log_probabilities, temperature_opt_state, temperature):
    """Mock temperature update function that returns predetermined values"""
    new_temperature = temperature * 0.95  # Decrease temperature a bit
    new_temperature_opt_state = temperature_opt_state  # We can't easily modify this
    temperature_loss = jnp.array(0.2)
    return new_temperature, new_temperature_opt_state, temperature_loss

def test_update_sac_basic(save_folder):
    """Test basic functionality of update_sac with simple inputs"""
    print("\n========== Update SAC: Basic Test ==========")
    
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
    buffer_weights = jnp.ones((batch_size, 1))  # equal weights
    
    # Initialize actor and critic parameters
    actor_params = actor.init(jax.random.fold_in(rng_key, 4), states)
    critic_params = critic.init(jax.random.fold_in(rng_key, 5), states, actions)
    critic_target_params = critic_params  # initially the same
    
    # Create optimizers
    actor_optimizer = optax.adam(learning_rate=3e-4)
    critic_optimizer = optax.adam(learning_rate=3e-4)
    temperature_optimizer = optax.adam(learning_rate=3e-4)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Initial temperature value and opt state
    temperature = 0.2
    temperature_opt_state = temperature_optimizer.init(jnp.log(temperature))
    
    # Random distributions for action sampling
    normal_dist_next = jax.random.normal(jax.random.fold_in(rng_key, 6), (batch_size, action_dim))
    normal_dist_actions = jax.random.normal(jax.random.fold_in(rng_key, 7), (batch_size, action_dim))
    
    # Set tau (target network update rate)
    tau = 0.005
    
    # Run test with first_step_bool = True (should skip temperature update)
    first_step_bool = True
    
    # Call update_sac
    (new_critic_params, new_critic_opt_state, critic_loss, td_errors, 
     new_actor_params, new_actor_opt_state, actor_loss, 
     new_temperature, new_temperature_opt_state, temperature_loss, 
     new_critic_target_params, current_log_probs, action_std) = update_sac(
        actor=actor,
        actor_params=actor_params,
        actor_opt_state=actor_opt_state,
        normal_distribution_for_next_actions=normal_dist_next,
        normal_distribution_for_actions=normal_dist_actions,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        buffer_weights=buffer_weights,
        temperature=temperature,
        temperature_opt_state=temperature_opt_state,
        critic_params=critic_params,
        critic_target_params=critic_target_params,
        critic_opt_state=critic_opt_state,
        critic_update_lambda=mock_critic_update,
        actor_update_lambda=mock_actor_update,
        temperature_update_lambda=mock_temperature_update,
        tau=tau,
        first_step_bool=first_step_bool
    )
    
    # Verify the function executes without errors
    test_basic_run_passed = True
    test_results.add_result(
        "Update SAC basic run",
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
    
    # Verify actor parameters were updated
    actor_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            actor_params, 
            new_actor_params
        )
    )
    test_results.add_result(
        "Actor parameters updated",
        actor_params_changed,
        "Expected actor parameters to change"
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
    
    # Verify temperature is unchanged on first step
    temp_unchanged = temperature == new_temperature
    test_results.add_result(
        "Temperature unchanged on first step",
        temp_unchanged,
        f"Expected temperature to be unchanged when first_step_bool=True, got {temperature} vs {new_temperature}"
    )
    
    # Verify temperature_loss is zero on first step
    temp_loss_zero = temperature_loss == 0.0
    test_results.add_result(
        "Temperature loss is zero on first step",
        temp_loss_zero,
        f"Expected temperature loss to be zero when first_step_bool=True, got {temperature_loss}"
    )
    
    # Run test with first_step_bool = False (should perform temperature update)
    first_step_bool = False
    
    # Call update_sac again
    (_, _, _, _, 
     _, _, _, 
     new_temperature2, _, temperature_loss2, 
     _, _, _) = update_sac(
        actor=actor,
        actor_params=new_actor_params,
        actor_opt_state=new_actor_opt_state,
        normal_distribution_for_next_actions=normal_dist_next,
        normal_distribution_for_actions=normal_dist_actions,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        buffer_weights=buffer_weights,
        temperature=new_temperature,
        temperature_opt_state=new_temperature_opt_state,
        critic_params=new_critic_params,
        critic_target_params=new_critic_target_params,
        critic_opt_state=new_critic_opt_state,
        critic_update_lambda=mock_critic_update,
        actor_update_lambda=mock_actor_update,
        temperature_update_lambda=mock_temperature_update,
        tau=tau,
        first_step_bool=first_step_bool
    )
    
    # Verify temperature is updated on second step
    temp_changed = new_temperature != new_temperature2
    test_results.add_result(
        "Temperature updated after first step",
        temp_changed,
        f"Expected temperature to change when first_step_bool=False, got {new_temperature} vs {new_temperature2}"
    )
    
    # Verify temperature_loss is non-zero on second step
    temp_loss_nonzero = temperature_loss2 != 0.0
    test_results.add_result(
        "Temperature loss is non-zero after first step",
        temp_loss_nonzero,
        f"Expected temperature loss to be non-zero when first_step_bool=False, got {temperature_loss2}"
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
        # We'll use Euclidean distance of a flattened parameter
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
        test_basic_run_passed, critic_params_changed, actor_params_changed,
        target_params_changed, temp_unchanged, temp_loss_zero,
        temp_changed, temp_loss_nonzero
    ])
    
    return all_basic_tests_passed

def test_update_sac_integration(save_folder):
    """Test the integration of all components in update_sac"""
    print("\n========== Update SAC: Integration Test ==========")
    
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
    
    # Test different buffer weight configurations
    uniform_weights = jnp.ones((batch_size, 1))
    # Create more diverse priotised weights that will produce a noticeably different loss
    prioritized_weights = jnp.linspace(0.1, 2.0, batch_size).reshape(batch_size, 1)
    zero_weights = jnp.zeros((batch_size, 1))
    
    # Initialize actor and critic parameters
    actor_params = actor.init(jax.random.fold_in(rng_key, 4), states)
    critic_params = critic.init(jax.random.fold_in(rng_key, 5), states, actions)
    critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
    
    # Create optimizers
    actor_optimizer = optax.adam(learning_rate=3e-4)
    critic_optimizer = optax.adam(learning_rate=3e-4)
    temperature_optimizer = optax.adam(learning_rate=3e-4)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Initial temperature and opt state
    temperature = 0.2
    temperature_opt_state = temperature_optimizer.init(jnp.log(temperature))
    
    # Random distributions for action sampling
    normal_dist_next = jax.random.normal(jax.random.fold_in(rng_key, 6), (batch_size, action_dim))
    normal_dist_actions = jax.random.normal(jax.random.fold_in(rng_key, 7), (batch_size, action_dim))
    
    # Set tau and other parameters
    tau = 0.005
    first_step_bool = False
    
    # Run update_sac with different buffer weights
    results = {}
    for name, weights in [
        ("Uniform", uniform_weights),
        ("Prioritized", prioritized_weights),
        ("Zero", zero_weights)
    ]:
        # Deep copy all parameters to avoid interference between tests
        actor_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), actor_params)
        critic_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
        critic_target_params_copy = jax.tree_util.tree_map(lambda x: x.copy(), critic_target_params)
        
        # Reset optimizer states
        actor_opt_state_copy = actor_optimizer.init(actor_params_copy)
        critic_opt_state_copy = critic_optimizer.init(critic_params_copy)
        temperature_copy = temperature
        temperature_opt_state_copy = temperature_optimizer.init(jnp.log(temperature_copy))
        
        # Call update_sac
        (new_critic_params, new_critic_opt_state, critic_loss, td_errors, 
         new_actor_params, new_actor_opt_state, actor_loss, 
         new_temperature, new_temperature_opt_state, temperature_loss, 
         new_critic_target_params, current_log_probs, action_std) = update_sac(
            actor=actor,
            actor_params=actor_params_copy,
            actor_opt_state=actor_opt_state_copy,
            normal_distribution_for_next_actions=normal_dist_next,
            normal_distribution_for_actions=normal_dist_actions,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            buffer_weights=weights,
            temperature=temperature_copy,
            temperature_opt_state=temperature_opt_state_copy,
            critic_params=critic_params_copy,
            critic_target_params=critic_target_params_copy,
            critic_opt_state=critic_opt_state_copy,
            critic_update_lambda=mock_critic_update,
            actor_update_lambda=mock_actor_update,
            temperature_update_lambda=mock_temperature_update,
            tau=tau,
            first_step_bool=first_step_bool
        )
        
        results[name] = {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "temperature_loss": float(temperature_loss),
            "temperature": float(new_temperature),
            "td_errors_mean": float(jnp.mean(td_errors)),
            "td_errors_std": float(jnp.std(td_errors)),
            "log_probs_mean": float(jnp.mean(current_log_probs)),
            "log_probs_std": float(jnp.std(current_log_probs)),
            "action_std_mean": float(jnp.mean(action_std)),
        }
    
    # Print results for debugging
    print("\nUpdate Results with Different Buffer Weights:")
    for name, metrics in results.items():
        print(f"\n{name} Weights:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
    
    # Verify that zero weights result in zero critic loss
    zero_loss_is_zero = results["Zero"]["critic_loss"] == 0.0
    test_results.add_result(
        "Zero weights produce zero critic loss",
        zero_loss_is_zero,
        f"Expected zero critic loss with zero weights, got {results['Zero']['critic_loss']}"
    )
    
    # Verify that uniform and priotised weights produce different losses
    uniform_vs_prioritized = results["Uniform"]["critic_loss"] != results["Prioritized"]["critic_loss"]
    test_results.add_result(
        "Different weights produce different critic losses",
        uniform_vs_prioritized,
        f"Expected different critic losses with different weights, got {results['Uniform']['critic_loss']} vs {results['Prioritized']['critic_loss']}"
    )
    
    # Verify actor loss is consistent across weight schemes (actor doesn't use weights)
    actor_loss_consistent = results["Uniform"]["actor_loss"] == results["Prioritized"]["actor_loss"] == results["Zero"]["actor_loss"]
    test_results.add_result(
        "Actor loss consistent across weight schemes",
        actor_loss_consistent,
        f"Expected consistent actor loss across weight schemes, got {results['Uniform']['actor_loss']} vs {results['Prioritized']['actor_loss']} vs {results['Zero']['actor_loss']}"
    )
    
    # Verify temperature loss is consistent across weight schemes (temperature doesn't use weights)
    temp_loss_consistent = results["Uniform"]["temperature_loss"] == results["Prioritized"]["temperature_loss"] == results["Zero"]["temperature_loss"]
    test_results.add_result(
        "Temperature loss consistent across weight schemes",
        temp_loss_consistent,
        f"Expected consistent temperature loss across weight schemes, got {results['Uniform']['temperature_loss']} vs {results['Prioritized']['temperature_loss']} vs {results['Zero']['temperature_loss']}"
    )
    
    # Plot comparison of metrics across weight schemes
    metrics_to_plot = ["critic_loss", "actor_loss", "temperature_loss", "td_errors_mean", "log_probs_mean"]
    
    plt.figure(figsize=(15, 10))
    bar_width = 0.25
    index = np.arange(len(metrics_to_plot))
    
    for i, (name, color) in enumerate([("Uniform", "blue"), ("Prioritized", "green"), ("Zero", "red")]):
        values = [results[name][metric] for metric in metrics_to_plot]
        plt.bar(index + i * bar_width, values, bar_width, color=color, label=f"{name} Weights")
    
    plt.title('Metrics Comparison Across Buffer Weight Schemes', fontsize=22)
    plt.xlabel('Metric', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.xticks(index + bar_width, metrics_to_plot, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/metrics_comparison_weights.png', dpi=300)
    
    all_integration_tests_passed = all([
        zero_loss_is_zero, uniform_vs_prioritized, 
        actor_loss_consistent, temp_loss_consistent
    ])
    
    return all_integration_tests_passed

def run_all_tests():
    """Run all update_sac verification tests"""
    print("\n===== STARTING UPDATE SAC VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/update_sac_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Basic update_sac functionality
    print("\n[1/2] Running Basic Update SAC Test...")
    basic_test_result = test_update_sac_basic(save_folder)
    
    # Test 2: Integration test with different buffer weights
    print("\n[2/2] Running Update SAC Integration Test...")
    integration_test_result = test_update_sac_integration(save_folder)
    
    print("\n===== UPDATE SAC VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/update_sac_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 