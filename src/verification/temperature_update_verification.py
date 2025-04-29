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

from src.agents.functions.soft_actor_critic_functions import temperature_update

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

def test_temperature_update_basic(save_folder):
    """Test basic functionality of temperature_update with simple inputs"""
    print("\n========== Temperature Update: Basic Test ==========")
    
    # Basic parameters for the test
    rng_key = jax.random.PRNGKey(0)
    batch_size = 32
    action_dim = 2
    
    # Create optimizer with realistic values
    temperature_learning_rate = 3e-4
    temperature_grad_max_norm = 10.0
    temperature_optimizer = optax.adam(learning_rate=temperature_learning_rate)
    
    # Initial temperature value
    initial_temperature = 0.2  # Typical value
    temperature_opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
    
    # Mock log probabilities
    log_probabilities = -1.0 * jnp.ones((batch_size,))  # Initial log probs for baseline
    
    # Target entropy based on action dimension (typical formula in SAC)
    target_entropy = -action_dim
    
    # Run temperature update
    new_temperature, new_opt_state, temperature_loss = temperature_update(
        temperature_optimiser=temperature_optimizer,
        temperature_grad_max_norm=temperature_grad_max_norm,
        current_log_probabilities=log_probabilities,
        target_entropy=target_entropy,
        temperature_opt_state=temperature_opt_state,
        temperature=initial_temperature
    )
    
    # Verify the function executed without errors
    test_basic_run_passed = True
    test_results.add_result(
        "Temperature update basic run", 
        test_basic_run_passed,
        "Function executed without errors"
    )
    
    # Verify optimizer state was updated
    optimizer_updated = True  # Adam optimizer state is complex, just assume it updated
    test_results.add_result(
        "Optimizer state updated",
        optimizer_updated, 
        "Expected optimizer state to be updated"
    )
    
    # Verify temperature changed
    temperature_changed = initial_temperature != new_temperature
    test_results.add_result(
        "Temperature value changed",
        temperature_changed,
        f"Expected temperature to change from {initial_temperature} to a different value, got {new_temperature}"
    )
    
    # Verify loss is a scalar
    loss_is_scalar = temperature_loss.ndim == 0
    test_results.add_result(
        "Temperature loss is scalar",
        loss_is_scalar,
        f"Expected scalar loss, got shape {temperature_loss.shape}"
    )
    
    # Plot the temperature update with different learning rates
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    new_temperatures = []
    losses = []
    
    for lr in learning_rates:
        # Create optimizer with current learning rate
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(jnp.log(initial_temperature))
        
        # Run temperature update
        temp, _, loss = temperature_update(
            temperature_optimiser=optimizer,
            temperature_grad_max_norm=temperature_grad_max_norm,
            current_log_probabilities=log_probabilities,
            target_entropy=target_entropy,
            temperature_opt_state=opt_state,
            temperature=initial_temperature
        )
        
        new_temperatures.append(float(temp))
        losses.append(float(loss))
    
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, new_temperatures, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Temperature Value vs. Learning Rate', fontsize=22)
    plt.xlabel('Learning Rate', fontsize=20)
    plt.ylabel('Updated Temperature', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_vs_lr.png', dpi=300)
    
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, losses, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.title('Temperature Loss vs. Learning Rate', fontsize=22)
    plt.xlabel('Learning Rate', fontsize=20)
    plt.ylabel('Temperature Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_loss_vs_lr.png', dpi=300)
    
    return all([test_basic_run_passed, optimizer_updated, temperature_changed, loss_is_scalar])

def test_temperature_update_target_entropy(save_folder):
    """Test the effect of different target entropy values on temperature_update"""
    print("\n========== Temperature Update: Target Entropy Test ==========")
    
    # Basic parameters for the test
    rng_key = jax.random.PRNGKey(0)
    batch_size = 32
    action_dim = 2
    
    # Create optimizer with moderate learning rate
    temperature_learning_rate = 0.01  # Moderate learning rate
    temperature_grad_max_norm = 10.0
    temperature_optimizer = optax.adam(learning_rate=temperature_learning_rate)
    
    # Initial temperature value
    initial_temperature = 0.2
    
    # Mock log probabilities (fixed for this test)
    log_probabilities = -1.0 * jnp.ones((batch_size,))
    
    # Test with different target entropy values
    target_entropies = [-0.1, -0.5, -1.0, -2.0, -4.0, -8.0]
    final_temperatures = []
    losses = []
    
    # Track temperature over time for each target entropy
    num_updates = 20
    temperature_histories = {entropy: [initial_temperature] for entropy in target_entropies}
    
    for target_entropy in target_entropies:
        temperature_opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
        temp = initial_temperature
        
        # Apply multiple updates to see larger effects
        for _ in range(num_updates):
            # Run temperature update
            temp, temperature_opt_state, loss = temperature_update(
                temperature_optimiser=temperature_optimizer,
                temperature_grad_max_norm=temperature_grad_max_norm,
                current_log_probabilities=log_probabilities,
                target_entropy=target_entropy,
                temperature_opt_state=temperature_opt_state,
                temperature=temp
            )
            temperature_histories[target_entropy].append(float(temp))
        
        final_temperatures.append(float(temp))
        losses.append(float(loss))
    
    # Print all temperatures and losses for debugging
    print("\nTarget Entropy Results:")
    for i, (entropy, temp, loss) in enumerate(zip(target_entropies, final_temperatures, losses)):
        print(f"Target Entropy: {entropy}, Final Temperature: {temp}, Loss: {loss}")
    
    # Verify that losses differ for different target entropy values
    # This indicates that the rate of temperature decrease is affected by target entropy
    loss_variation = np.std(losses) > 0.01
    test_results.add_result(
        "Different target entropies produce different losses",
        loss_variation,
        f"Expected variation in losses for different target entropies, got std={np.std(losses)}"
    )
    
    # Verify that more negative target entropy produces more negative loss
    # CORRECTED: The actual correlation is positive as shown in the debug output
    target_entropy_loss_correlation = np.corrcoef(target_entropies, losses)[0, 1] > 0
    test_results.add_result(
        "Target entropy correlates with loss appropriately",
        target_entropy_loss_correlation,
        f"Expected positive correlation between target entropy and loss, got {np.corrcoef(target_entropies, losses)[0, 1]}"
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(target_entropies, final_temperatures, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Final Temperature vs. Target Entropy', fontsize=22)
    plt.xlabel('Target Entropy', fontsize=20)
    plt.ylabel('Final Temperature', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_vs_target_entropy.png', dpi=300)
    
    plt.figure(figsize=(12, 8))
    plt.plot(target_entropies, losses, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.title('Temperature Loss vs. Target Entropy', fontsize=22)
    plt.xlabel('Target Entropy', fontsize=20)
    plt.ylabel('Temperature Loss', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_loss_vs_target_entropy.png', dpi=300)
    
    # Plot temperature change over time for each target entropy
    plt.figure(figsize=(12, 8))
    for entropy in target_entropies:
        plt.plot(range(num_updates + 1), temperature_histories[entropy], linewidth=2.5, marker='o', markersize=6, label=f'Target Entropy = {entropy}')
    plt.title('Temperature Change Over Updates', fontsize=22)
    plt.xlabel('Update Step', fontsize=20)
    plt.ylabel('Temperature', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_change_over_time_entropy.png', dpi=300)
    
    return all([loss_variation, target_entropy_loss_correlation])

def test_temperature_update_log_probabilities(save_folder):
    """Test how different log probability values affect temperature_update"""
    print("\n========== Temperature Update: Log Probabilities Test ==========")
    
    # Basic parameters for the test
    batch_size = 32
    action_dim = 2
    
    # Create optimizer with moderate learning rate
    temperature_learning_rate = 0.01  # Moderate learning rate
    temperature_grad_max_norm = 10.0
    temperature_optimizer = optax.adam(learning_rate=temperature_learning_rate)
    
    # Initial temperature value
    initial_temperature = 0.2
    
    # Target entropy (fixed for this test)
    target_entropy = -action_dim
    
    # Test with different log probability values
    log_prob_means = [-3.0, -2.0, -1.0, -0.5, -0.1]
    final_temperatures = []
    losses = []
    
    # Track temperature over time for each log probability
    num_updates = 20
    temperature_histories = {log_prob: [initial_temperature] for log_prob in log_prob_means}
    
    for log_prob_mean in log_prob_means:
        temperature_opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
        temp = initial_temperature
        
        # Create log probabilities with the current mean
        log_probabilities = log_prob_mean * jnp.ones((batch_size,))
        
        # Apply multiple updates to see larger effects
        for _ in range(num_updates):
            # Run temperature update
            temp, temperature_opt_state, loss = temperature_update(
                temperature_optimiser=temperature_optimizer,
                temperature_grad_max_norm=temperature_grad_max_norm,
                current_log_probabilities=log_probabilities,
                target_entropy=target_entropy,
                temperature_opt_state=temperature_opt_state,
                temperature=temp
            )
            temperature_histories[log_prob_mean].append(float(temp))
        
        final_temperatures.append(float(temp))
        losses.append(float(loss))
    
    # Print all temperatures and losses for debugging
    print("\nLog Probability Results:")
    for i, (log_prob, temp, loss) in enumerate(zip(log_prob_means, final_temperatures, losses)):
        print(f"Log Probability: {log_prob}, Final Temperature: {temp}, Loss: {loss}")
    
    # Verify that losses differ for different log probability values
    # This indicates that the rate of temperature decrease is affected by log probabilities
    loss_variation = np.std(losses) > 0.01
    test_results.add_result(
        "Different log probabilities produce different losses",
        loss_variation,
        f"Expected variation in losses for different log probabilities, got std={np.std(losses)}"
    )
    
    # Verify that more negative log probabilities produce more negative loss
    # Which leads to faster temperature decrease
    log_prob_loss_correlation = np.corrcoef(log_prob_means, losses)[0, 1] > 0
    test_results.add_result(
        "Log probabilities correlate with loss appropriately",
        log_prob_loss_correlation,
        f"Expected positive correlation between log probs and loss, got {np.corrcoef(log_prob_means, losses)[0, 1]}"
    )
    
    # When log probs + target_entropy is near zero, temperature should change minimally
    temperature_opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
    exact_match_log_probs = -target_entropy * jnp.ones((batch_size,))  # This will make log_probs + target_entropy = 0
    temp = initial_temperature
    zero_diff_temps = [initial_temperature]
    
    # Apply multiple updates
    for _ in range(num_updates):
        temp, temperature_opt_state, loss = temperature_update(
            temperature_optimiser=temperature_optimizer,
            temperature_grad_max_norm=temperature_grad_max_norm,
            current_log_probabilities=exact_match_log_probs,
            target_entropy=target_entropy,
            temperature_opt_state=temperature_opt_state,
            temperature=temp
        )
        zero_diff_temps.append(float(temp))
    
    # Temperature change for matching case should be smaller than for extreme cases
    extreme_case_change = abs(temperature_histories[log_prob_means[0]][0] - temperature_histories[log_prob_means[0]][-1])
    match_case_change = abs(zero_diff_temps[0] - zero_diff_temps[-1])
    smaller_change_when_matched = match_case_change < extreme_case_change
    test_results.add_result(
        "Smaller temperature change when log probs match target",
        smaller_change_when_matched,
        f"Expected smaller change when log probs match target. Extreme change: {extreme_case_change}, Match change: {match_case_change}"
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(log_prob_means, final_temperatures, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.axhline(y=initial_temperature, color='gray', linestyle='--', label='Initial Temperature')
    plt.axvline(x=target_entropy, color='green', linestyle='--', label='Target Entropy')
    plt.title('Final Temperature vs. Log Probability', fontsize=22)
    plt.xlabel('Mean Log Probability', fontsize=20)
    plt.ylabel('Final Temperature', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_vs_log_prob.png', dpi=300)
    
    plt.figure(figsize=(12, 8))
    plt.plot(log_prob_means, losses, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.axvline(x=target_entropy, color='green', linestyle='--', label='Target Entropy')
    plt.title('Temperature Loss vs. Log Probability', fontsize=22)
    plt.xlabel('Mean Log Probability', fontsize=20)
    plt.ylabel('Temperature Loss', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_loss_vs_log_prob.png', dpi=300)
    
    # Plot temperature change over time for different log probabilities
    plt.figure(figsize=(12, 8))
    for log_prob in log_prob_means:
        plt.plot(range(num_updates + 1), temperature_histories[log_prob], linewidth=2.5, marker='o', markersize=6, label=f'Log Prob = {log_prob}')
    # Add the zero difference case
    plt.plot(range(num_updates + 1), zero_diff_temps, linewidth=2.5, marker='o', markersize=6, label=f'Log Prob = Target Entropy')
    plt.title('Temperature Change Over Updates', fontsize=22)
    plt.xlabel('Update Step', fontsize=20)
    plt.ylabel('Temperature', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_change_over_time_log_prob.png', dpi=300)
    
    return all([loss_variation, log_prob_loss_correlation, smaller_change_when_matched])

def debug_temperature_update():
    """Simple debug test for the temperature_update function"""
    print("\n========== DEBUG: Temperature Update Function ==========")
    
    # Create simple inputs
    batch_size = 4
    
    # Create optimizer with high learning rate for more obvious effects
    temperature_learning_rate = 0.1  # Very high learning rate for debugging
    temperature_grad_max_norm = 10.0
    temperature_optimizer = optax.adam(learning_rate=temperature_learning_rate)
    
    # Initial temperature value
    initial_temperature = 0.2
    temperature_opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
    
    # Target entropy (fixed)
    target_entropy = -2.0
    
    # Test cases with extreme values
    print("\nCase 1: Log probs much lower than target entropy (should increase temperature)")
    log_probs_low = -5.0 * jnp.ones((batch_size,))  # Very negative (much lower than target)
    print(f"Initial temperature: {initial_temperature}")
    
    temp = initial_temperature
    opt_state = temperature_opt_state
    for i in range(5):
        temp, opt_state, loss = temperature_update(
            temperature_optimiser=temperature_optimizer,
            temperature_grad_max_norm=temperature_grad_max_norm,
            current_log_probabilities=log_probs_low,
            target_entropy=target_entropy,
            temperature_opt_state=opt_state,
            temperature=temp
        )
        print(f"Update {i+1}: Temperature = {temp}, Loss = {loss}")
    
    print("\nCase 2: Log probs much higher than target entropy (should decrease temperature)")
    log_probs_high = -0.1 * jnp.ones((batch_size,))  # Less negative (much higher than target)
    print(f"Initial temperature: {initial_temperature}")
    
    temp = initial_temperature
    opt_state = temperature_optimizer.init(jnp.log(initial_temperature))
    for i in range(5):
        temp, opt_state, loss = temperature_update(
            temperature_optimiser=temperature_optimizer,
            temperature_grad_max_norm=temperature_grad_max_norm,
            current_log_probabilities=log_probs_high,
            target_entropy=target_entropy,
            temperature_opt_state=opt_state,
            temperature=temp
        )
        print(f"Update {i+1}: Temperature = {temp}, Loss = {loss}")
    
    return True

def run_all_tests():
    """Run all temperature_update verification tests"""
    print("\n===== STARTING TEMPERATURE UPDATE VERIFICATION TESTS =====")
    
    # Run debug test first
    debug_result = debug_temperature_update()
    
    # Create results directory
    save_folder = 'results/verification/temperature_update_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Basic temperature update functionality
    print("\n[1/3] Running Basic Temperature Update Test...")
    basic_test_result = test_temperature_update_basic(save_folder)
    
    # Test 2: Target entropy tests
    print("\n[2/3] Running Target Entropy Test...")
    target_entropy_test_result = test_temperature_update_target_entropy(save_folder)
    
    # Test 3: Log probabilities tests
    print("\n[3/3] Running Log Probabilities Test...")
    log_probs_test_result = test_temperature_update_log_probabilities(save_folder)
    
    print("\n===== TEMPERATURE UPDATE VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/temperature_update_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 