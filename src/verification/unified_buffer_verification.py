import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import time
import csv
import datetime

# Add src to path
sys.path.append('.')

from src.agents.functions.buffers import PERBuffer, compute_n_step_single

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

def test_per_buffer_regular_vs_uniform(save_folder):
    """Test PER buffer in both prioritised and uniform sampling modes"""
    print("\n========== PER Buffer: Regular vs Uniform Test ==========")
    state_dim = 10
    action_dim = 3
    buffer_size = 1000
    batch_size = 32
    gamma = 0.99
    alpha = 0.6  
    beta = 0.4
    beta_decay = 1.0  # No decay for testing
    trajectory_length = 10

    # Create a regular PER buffer
    per_buffer = PERBuffer(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        beta_decay=beta_decay,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    per_buffer.set_uniform_sampling(False)  # Use prioritised sampling
    
    # Create a uniform PER buffer
    uniform_buffer = PERBuffer(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        beta_decay=beta_decay,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    uniform_buffer.set_uniform_sampling(True)  # Use uniform sampling

    # Fill buffers with same random data
    rng_key = jax.random.PRNGKey(42)
    keys = jax.random.split(rng_key, buffer_size)
    
    # Add dummy TD errors
    td_errors = jnp.power(10, jnp.linspace(0, 3, buffer_size))  # 1 to 1000 with exponential distribution
    
    for i in range(buffer_size):
        state = jax.random.normal(keys[i], (state_dim,))
        action = jax.random.normal(jax.random.fold_in(keys[i], 1), (action_dim,))
        reward = jnp.array([jax.random.uniform(jax.random.fold_in(keys[i], 2))])
        next_state = jax.random.normal(jax.random.fold_in(keys[i], 3), (state_dim,))
        done = jnp.array([jax.random.uniform(jax.random.fold_in(keys[i], 4)) > 0.8])
        
        # Add data to buffers directly without reassignment
        per_buffer.add(state, action, reward[0], next_state, done[0], td_errors[i])
        uniform_buffer.add(state, action, reward[0], next_state, done[0], td_errors[i])
    
    # Update priorities explicitly to ensure they match
    per_buffer.update_priorities(jnp.arange(buffer_size), td_errors)
    uniform_buffer.update_priorities(jnp.arange(buffer_size), td_errors)
    
    # Sample from both buffers
    rng_key = jax.random.PRNGKey(99)
    sample_keys = jax.random.split(rng_key, 100)
    
    # Collect samples from both buffers
    per_indices_list = []
    per_weights_list = []
    uniform_indices_list = []
    uniform_weights_list = []
    
    num_samples = 50
    for i in range(num_samples):
        _, _, _, _, _, per_indices, per_weights = per_buffer(sample_keys[i])
        _, _, _, _, _, uniform_indices, uniform_weights = uniform_buffer(sample_keys[i])
        
        per_indices_list.append(per_indices)
        per_weights_list.append(per_weights)
        uniform_indices_list.append(uniform_indices)
        uniform_weights_list.append(uniform_weights)
    
    per_indices_all = jnp.concatenate(per_indices_list)
    per_weights_all = jnp.concatenate(per_weights_list)
    uniform_indices_all = jnp.concatenate(uniform_indices_list)
    uniform_weights_all = jnp.concatenate(uniform_weights_list)
    
    # Analysis and plotting
    print("\n----- PER Buffer Test Results -----")
    print(f"Buffer size: {buffer_size}, Alpha: {alpha}, Beta: {beta}")
    
    print("\n== Prioritized Sampling ==")
    print(f"Weight range: min={per_weights_all.min():.6f}, max={per_weights_all.max():.6f}")
    print(f"Weight variety: {jnp.unique(per_weights_all).size} unique values out of {per_weights_all.size} samples")
    
    print("\n== Uniform Sampling ==")
    print(f"Weight range: min={uniform_weights_all.min():.6f}, max={uniform_weights_all.max():.6f}")
    print(f"Weight variety: {jnp.unique(uniform_weights_all).size} unique values out of {uniform_weights_all.size} samples")
    
    # Analyze index distribution
    high_priority_indices = jnp.arange(buffer_size - 100, buffer_size)  # Top 100 priorities
    per_high_freq = jnp.sum(jnp.isin(per_indices_all, high_priority_indices)) / per_indices_all.size
    uniform_high_freq = jnp.sum(jnp.isin(uniform_indices_all, high_priority_indices)) / uniform_indices_all.size
    
    print("\n== Sampling Frequency ==")
    print(f"PER: Frequency of high-priority indices: {per_high_freq:.4f}")
    print(f"Uniform: Frequency of high-priority indices: {uniform_high_freq:.4f}")
    print(f"Ratio (PER/Uniform): {per_high_freq/uniform_high_freq if uniform_high_freq > 0 else 'inf':.2f}x")
    
    # Create plots folder if it doesn't exist
    Path("plots").mkdir(exist_ok=True)
    
    # Plot weight distribution
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(np.array(td_errors), bins=30, alpha=0.7)
    ax1.set_title('Priority Distribution in Buffer', fontsize=24)
    ax1.set_yscale('log')
    ax1.set_xlabel('Priority', fontsize=20)
    ax1.set_ylabel('Count (log scale)', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(np.array(per_weights_all), bins=30, alpha=0.7, color='green')
    ax2.set_title('PER Sampling Weights', fontsize=24)
    ax2.set_xlabel('Weight', fontsize=20)
    ax2.set_ylabel('Count', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True)
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.hist(np.array(uniform_weights_all), bins=30, alpha=0.7, color='red')
    ax3.set_title('Uniform Sampling Weights', fontsize=24)
    ax3.set_xlabel('Weight', fontsize=20)
    ax3.set_ylabel('Count', fontsize=20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.grid(True)
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.hist(np.array(per_indices_all), bins=30, alpha=0.5, color='blue', label='PER')
    ax4.hist(np.array(uniform_indices_all), bins=30, alpha=0.5, color='orange', label='Uniform')
    ax4.set_title('Sampled Indices Comparison', fontsize=24)
    ax4.set_xlabel('Index', fontsize=20)
    ax4.set_ylabel('Frequency', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)
    ax4.legend(fontsize=20)
    plt.savefig(f'{save_folder}/per_buffer_test.png')
    print(f"\nPlot saved to {save_folder}/per_buffer_test.png")
    
    # Verification checks
    # Check 1: Uniform weights should all be 1.0
    uniform_weights_all_ones = jnp.allclose(uniform_weights_all, 1.0, rtol=1e-5)
    test_results.add_result(
        "Uniform sampling weights all equal to 1.0", 
        uniform_weights_all_ones,
        f"Expected all weights to be 1.0, but found range: {uniform_weights_all.min():.6f} to {uniform_weights_all.max():.6f}"
    )
    
    # Check 2: PER weights should have variety
    per_weights_variety = jnp.unique(per_weights_all).size > buffer_size / 10
    test_results.add_result(
        "PER weights have sufficient variety",
        per_weights_variety,
        f"Found only {jnp.unique(per_weights_all).size} unique weights out of {per_weights_all.size} samples"
    )
    
    # Check 3: PER should sample high-priority items more often than uniform
    sampling_bias_correct = per_high_freq > uniform_high_freq * 1.2
    test_results.add_result(
        "PER samples high-priority items more frequently",
        sampling_bias_correct,
        f"PER high-priority frequency ({per_high_freq:.4f}) not significantly higher than uniform ({uniform_high_freq:.4f})"
    )
    
    return all([uniform_weights_all_ones, per_weights_variety, sampling_bias_correct])

def test_extreme_priorities(save_folder):
    """Test PER buffer with extreme priority differences to verify operation"""
    print("\n========== PER Buffer: Extreme Priorities Test ==========")
    state_dim = 4
    action_dim = 2
    buffer_size = 100
    batch_size = 10
    alpha = 0.6
    beta = 0.4
    gamma = 0.99
    trajectory_length = 5
    beta_decay = 1.0
    
    # Create buffer
    buffer = PERBuffer(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        beta_decay=beta_decay,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    buffer.set_uniform_sampling(False)  # Ensure prioritised sampling
    
    # Fill buffer with basic data
    rng_key = jax.random.PRNGKey(42)
    keys = jax.random.split(rng_key, buffer_size)
    
    for i in range(buffer_size):
        state = jax.random.normal(keys[i], (state_dim,))
        action = jax.random.normal(jax.random.fold_in(keys[i], 1), (action_dim,))
        reward = jnp.array([jax.random.uniform(jax.random.fold_in(keys[i], 2))])
        next_state = jax.random.normal(jax.random.fold_in(keys[i], 3), (state_dim,))
        done = jnp.array([0.0])
        td_error = 1.0
        
        # Add directly without reassignment
        buffer.add(state, action, reward[0], next_state, done[0], td_error)
    
    # Set extreme priorities
    # One entry has 1000x higher priority than all others
    priorities = jnp.ones(buffer_size)
    priorities = priorities.at[50].set(1000.0)  # Make index 50 have extreme priority
    buffer.update_priorities(jnp.arange(buffer_size), priorities)
    
    # Sample multiple times to see if high-priority item is sampled frequently
    sample_count = jnp.zeros(buffer_size)
    num_samples = 1000
    
    rng_key = jax.random.PRNGKey(99)
    sample_keys = jax.random.split(rng_key, num_samples)
    
    for i in range(num_samples):
        _, _, _, _, _, indices, weights = buffer(sample_keys[i])
        # Count occurrences of each index
        for idx in indices:
            sample_count = sample_count.at[idx].add(1)
    
    print("\n----- Extreme Priority Test -----")
    print(f"Index 50 priority: 1000.0, All others: 1.0")
    print(f"Index 50 was sampled {sample_count[50]} times out of {num_samples*batch_size} samples")
    print(f"Expected frequency with alpha={alpha}: {(1000.0**alpha)/((1000.0**alpha) + (buffer_size-1)):.4f}")
    actual_frequency = sample_count[50]/(num_samples*batch_size)
    print(f"Actual frequency: {actual_frequency:.4f}")
    
    # Plot sampling distribution
    plt.figure(figsize=(20, 15))
    plt.bar(np.arange(buffer_size), np.array(sample_count), alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', label='High Priority Index')
    plt.title('Sampling Distribution with Extreme Priority', fontsize = 22)
    plt.xlabel('Buffer Index', fontsize=20)
    plt.ylabel('Sample Count', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{save_folder}/extreme_priority_test.png')
    print(f"Plot saved to {save_folder}/extreme_priority_test.png")
    
    # Verification checks
    # Calculate expected frequency
    expected_freq = (1000.0**alpha)/((1000.0**alpha) + (buffer_size-1))
    
    # Check if actual frequency is reasonably close to expected
    # Allow for some sampling variance 
    frequency_accurate = abs(actual_frequency - expected_freq) / expected_freq < 0.15  # Within 15% of expected
    test_results.add_result(
        "Extreme priority sampling frequency matches expectation",
        frequency_accurate,
        f"Expected frequency {expected_freq:.4f}, actual {actual_frequency:.4f}, difference: {abs(actual_frequency - expected_freq) / expected_freq:.2%}"
    )
    
    # Check if high-priority item is sampled significantly more than others
    high_priority_dominates = sample_count[50] > 5 * jnp.mean(sample_count[sample_count != sample_count[50]])
    test_results.add_result(
        "High-priority item dominates sampling",
        high_priority_dominates,
        "High-priority item not sampled significantly more often than others"
    )
    
    return all([frequency_accurate, high_priority_dominates])

def test_n_step_rewards():
    """Test n-step reward computation in buffers"""
    print("\n========== N-Step Reward Computation Test ==========")
    gamma = 0.99
    state_dim = 4
    action_dim = 2
    n_step = 3
    
    # Create a sample trajectory
    # (state, action, reward, next_state, done)
    trajectory = [
        (jnp.ones(state_dim), jnp.ones(action_dim), 1.0, jnp.ones(state_dim) * 2, 0.0),  # Step 1
        (jnp.ones(state_dim) * 2, jnp.ones(action_dim) * 2, 2.0, jnp.ones(state_dim) * 3, 0.0),  # Step 2
        (jnp.ones(state_dim) * 3, jnp.ones(action_dim) * 3, 3.0, jnp.ones(state_dim) * 4, 0.0),  # Step 3
        (jnp.ones(state_dim) * 4, jnp.ones(action_dim) * 4, 4.0, jnp.ones(state_dim) * 5, 0.0),  # Step 4
        (jnp.ones(state_dim) * 5, jnp.ones(action_dim) * 5, 5.0, jnp.ones(state_dim) * 6, 1.0),  # Step 5 (terminal)
    ]
    
    # Convert trajectory to buffer format
    buffer_data = []
    for state, action, reward, next_state, done in trajectory:
        buffer_data.append(jnp.concatenate([state, action, jnp.array([reward]), next_state, jnp.array([done])]))
    
    buffer_data = jnp.stack(buffer_data)
    
    # Call n-step computation directly for demonstration
    print("\nManual verification of n-step rewards:")
    all_matches = True
    for i in range(len(trajectory)):
        # Consider only the next n transitions (or fewer if near the end)
        n_step_buffer = buffer_data[i:i+n_step]
        if len(n_step_buffer) < n_step:
            # Pad with zeros if needed
            padding = jnp.zeros((n_step - len(n_step_buffer), buffer_data.shape[1]))
            n_step_buffer = jnp.concatenate([n_step_buffer, padding])
            
        G, next_s, done = compute_n_step_single(n_step_buffer, gamma, state_dim, action_dim, n_step)
        
        # Calculate expected n-step return manually for verification
        expected_G = 0
        done_found = False
        for j in range(min(n_step, len(trajectory) - i)):
            if done_found:
                break
            _, _, r, _, d = trajectory[i + j]
            expected_G += (gamma ** j) * r
            if d > 0.5:  # Terminal state
                done_found = True
                
        # For step 3, we need to properly account for the terminal state at step 5
        # This is where we get R3 + gamma*R4 + gamma^2*R5 but with zero future reward after R5
        if i == 2:  # Step 3
            expected_G = 3.0 + gamma * 4.0 + gamma**2 * 5.0
        # For step 4, we get R4 + gamma*R5 with zero future reward after R5
        elif i == 3:  # Step 4
            expected_G = 4.0 + gamma * 5.0
                
        print(f"Step {i+1}:")
        print(f"  Computed n-step return: {G:.4f}")
        print(f"  Expected n-step return: {expected_G:.4f}")
        matches = jnp.isclose(G, expected_G, rtol=1e-5)
        print(f"  Matches: {matches}")
        print(f"  Terminal found: {done > 0.5}")
        
        if not matches:
            all_matches = False
    
    # Test in a full buffer context with PER buffer
    print("\nTesting n-step rewards in full buffer context:")
    buffer = PERBuffer(
        gamma=gamma,
        alpha=0.6,
        beta=0.4,
        beta_decay=1.0,
        buffer_size=100,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=n_step,
        batch_size=32
    )
    
    # Fill buffer with the trajectory data
    for state, action, reward, next_state, done in trajectory:
        buffer.add(state, action, reward, next_state, done, 1.0)  # Use dummy TD error of 1.0
    
    # Sample from buffer and verify rewards
    rng_key = jax.random.PRNGKey(42)
    states, actions, rewards, next_states, dones, _, _ = buffer(rng_key)
    
    print(f"  Buffer contains {len(buffer)} transitions")
    print(f"  Sample rewards: {rewards.flatten()}")
    
    # Verification checks
    test_results.add_result(
        "N-step reward calculations match expected values",
        all_matches,
        "At least one n-step reward calculation did not match expected value"
    )
    
    # Check that buffer contains correct number of transitions
    buffer_correct_size = len(buffer) == len(trajectory)
    test_results.add_result(
        "Buffer contains correct number of transitions",
        buffer_correct_size,
        f"Expected {len(trajectory)} transitions, found {len(buffer)}"
    )
    
    return all_matches and buffer_correct_size


def test_buffer_performance():
    """Test buffer performance with different sampling methods"""
    print("\n========== Buffer Performance Test ==========")
    state_dim = 32
    action_dim = 8
    buffer_size = 50000
    batch_size = 256
    gamma = 0.99
    alpha = 0.6
    beta = 0.4
    trajectory_length = 5
    
    # Create different types of buffers
    per_buffer = PERBuffer(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        beta_decay=1.0,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    per_buffer.set_uniform_sampling(False)
    
    uniform_buffer = PERBuffer(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        beta_decay=1.0,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    uniform_buffer.set_uniform_sampling(True)
    
    # Test data insertion performance
    print("\nTesting insertion performance (1000 items):")
    
    rng_key = jax.random.PRNGKey(42)
    keys = jax.random.split(rng_key, 1000)
    
    # Prepare data for insertion
    insertion_data = []
    for i in range(1000):
        state = jax.random.normal(keys[i], (state_dim,))
        action = jax.random.normal(jax.random.fold_in(keys[i], 1), (action_dim,))
        reward = jnp.array(jax.random.uniform(jax.random.fold_in(keys[i], 2)))
        next_state = jax.random.normal(jax.random.fold_in(keys[i], 3), (state_dim,))
        done = jnp.array(jax.random.uniform(jax.random.fold_in(keys[i], 4)) > 0.9)
        td_error = jnp.array(jax.random.uniform(jax.random.fold_in(keys[i], 5)))
        insertion_data.append((state, action, reward, next_state, done, td_error))
    
    # PER buffer insertion
    start_time = time.time()
    for state, action, reward, next_state, done, td_error in insertion_data:
        per_buffer.add(state, action, reward, next_state, done, td_error)
    per_time = time.time() - start_time
    
    # Uniform buffer insertion
    start_time = time.time()
    for state, action, reward, next_state, done, td_error in insertion_data:
        uniform_buffer.add(state, action, reward, next_state, done, td_error)
    uniform_time = time.time() - start_time
    
    print(f"  PER Buffer insertion time: {per_time:.4f} seconds")
    print(f"  Uniform Buffer insertion time: {uniform_time:.4f} seconds")
    
    # Test sampling performance
    print("\nTesting sampling performance (100 batches):")
    
    rng_key = jax.random.PRNGKey(99)
    sample_keys = jax.random.split(rng_key, 100)
    
    # PER buffer sampling
    start_time = time.time()
    for i in range(100):
        _ = per_buffer(sample_keys[i])
    per_time = time.time() - start_time
    
    # Uniform buffer sampling
    start_time = time.time()
    for i in range(100):
        _ = uniform_buffer(sample_keys[i])
    uniform_time = time.time() - start_time
    
    print(f"  PER Buffer sampling time: {per_time:.4f} seconds")
    print(f"  Uniform Buffer sampling time: {uniform_time:.4f} seconds")
    print(f"  Speedup ratio: {per_time/uniform_time:.2f}x")
    
    # Verification checks
    # Both buffers should complete operations without errors
    test_results.add_result(
        "Buffer insertion successful",
        True,  # Already true if we got this far
        "Buffer insertion failed"
    )
    
    test_results.add_result(
        "Buffer sampling successful",
        True,  # Already true if we got this far
        "Buffer sampling failed"
    )
    
    # Performance is not a pass/fail metric, just informational
    
    return True

def run_all_tests():
    """Run all buffer verification tests"""
    print("\n===== STARTING UNIFIED BUFFER VERIFICATION TESTS =====")
    
    # Create plots directory
    save_folder = 'results/verification/unified_buffer_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test standard buffer vs PER buffer
    print("\n[1/4] Running PER vs Uniform Test...")
    per_uniform_result = test_per_buffer_regular_vs_uniform(save_folder)
    
    # Test extreme priorities
    print("\n[2/4] Running Extreme Priorities Test...")
    extreme_result = test_extreme_priorities(save_folder)
    
    # Test n-step rewards computation
    print("\n[3/4] Running N-Step Rewards Test...")
    n_step_result = test_n_step_rewards()
    
    # Test buffer performance
    print("\n[4/4] Running Buffer Performance Test...")
    performance_result = test_buffer_performance()
    
    print("\n===== BUFFER VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"results/verification/unified_buffer_verification/buffer_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 