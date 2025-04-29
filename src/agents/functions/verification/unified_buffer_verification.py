import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from functools import partial
import time

# Add src to path
sys.path.append('.')

from src.agents.functions.buffers import PERBuffer, ReplayBuffer, compute_n_step_single

def test_per_buffer_regular_vs_uniform():
    """Test PER buffer in both prioritized and uniform sampling modes"""
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
    per_buffer.set_uniform_sampling(False)  # Use prioritized sampling
    
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
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(np.array(td_errors), bins=30, alpha=0.7)
    plt.title('Priority Distribution in Buffer')
    plt.yscale('log')
    plt.xlabel('Priority')
    plt.ylabel('Count (log scale)')
    
    plt.subplot(2, 2, 2)
    plt.hist(np.array(per_weights_all), bins=30, alpha=0.7, color='green')
    plt.title('PER Sampling Weights')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.hist(np.array(uniform_weights_all), bins=30, alpha=0.7, color='red')
    plt.title('Uniform Sampling Weights')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 4)
    plt.hist(np.array(per_indices_all), bins=30, alpha=0.5, color='blue', label='PER')
    plt.hist(np.array(uniform_indices_all), bins=30, alpha=0.5, color='orange', label='Uniform')
    plt.title('Sampled Indices Comparison')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/per_buffer_test.png')
    print("\nPlot saved to plots/per_buffer_test.png")

def test_extreme_priorities():
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
    buffer.set_uniform_sampling(False)  # Ensure prioritized sampling
    
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
    print(f"Actual frequency: {sample_count[50]/(num_samples*batch_size):.4f}")
    
    # Plot sampling distribution
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(buffer_size), np.array(sample_count), alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', label='High Priority Index')
    plt.title('Sampling Distribution with Extreme Priority')
    plt.xlabel('Buffer Index')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.savefig('plots/extreme_priority_test.png')
    print("Plot saved to plots/extreme_priority_test.png")

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
        # Format: state, action, reward, next_state, done
        (jnp.ones(state_dim), jnp.ones(action_dim), 1.0, jnp.ones(state_dim) * 2, 0.0),  # Step 1
        (jnp.ones(state_dim) * 2, jnp.ones(action_dim) * 2, 2.0, jnp.ones(state_dim) * 3, 0.0),  # Step 2
        (jnp.ones(state_dim) * 3, jnp.ones(action_dim) * 3, 3.0, jnp.ones(state_dim) * 4, 0.0),  # Step 3
        (jnp.ones(state_dim) * 4, jnp.ones(action_dim) * 4, 4.0, jnp.ones(state_dim) * 5, 0.0),  # Step 4
        (jnp.ones(state_dim) * 5, jnp.ones(action_dim) * 5, 5.0, jnp.ones(state_dim) * 6, 1.0),  # Step 5 (terminal)
    ]
    
    # Convert trajectory to buffer format
    buffer_data = []
    for state, action, reward, next_state, done in trajectory:
        buffer_data.append(jnp.concatenate([
            state, action, jnp.array([reward]), next_state, jnp.array([done])
        ]))
    
    buffer_data = jnp.stack(buffer_data)
    
    # Call n-step computation directly for demonstration
    print("\nManual verification of n-step rewards:")
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
                
        print(f"Step {i+1}:")
        print(f"  Computed n-step return: {G:.4f}")
        print(f"  Expected n-step return: {expected_G:.4f}")
        print(f"  Matches: {jnp.isclose(G, expected_G)}")
        print(f"  Terminal found: {done > 0.5}")
    
    # Test in a full buffer context
    print("\nTesting n-step rewards in full buffer context:")
    buffer = ReplayBuffer(
        gamma=gamma,
        buffer_size=100,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=n_step,
        batch_size=32
    )
    
    # Fill buffer with the trajectory data
    for state, action, reward, next_state, done in trajectory:
        buffer.add(state, action, reward, next_state, done, 0.0)  # TD error not used in regular buffer
    
    # Sample from buffer and verify rewards
    rng_key = jax.random.PRNGKey(42)
    states, actions, rewards, next_states, dones, _, _ = buffer(rng_key)
    
    print(f"  Buffer contains {len(buffer)} transitions")
    print(f"  Sample rewards: {rewards.flatten()}")
    
    # Compare with manually calculated n-step rewards
    if len(buffer) > 0:
        print("  Note: Rewards from buffer should reflect n-step returns")

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
    
    regular_buffer = ReplayBuffer(
        gamma=gamma,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=trajectory_length,
        batch_size=batch_size
    )
    
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
    
    # Regular buffer insertion
    start_time = time.time()
    for state, action, reward, next_state, done, td_error in insertion_data:
        regular_buffer.add(state, action, reward, next_state, done, td_error)
    regular_time = time.time() - start_time
    
    print(f"  PER Buffer insertion time: {per_time:.4f} seconds")
    print(f"  Uniform Buffer insertion time: {uniform_time:.4f} seconds")
    print(f"  Regular Buffer insertion time: {regular_time:.4f} seconds")
    
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
    
    # Regular buffer sampling
    start_time = time.time()
    for i in range(100):
        _ = regular_buffer(sample_keys[i])
    regular_time = time.time() - start_time
    
    print(f"  PER Buffer sampling time: {per_time:.4f} seconds")
    print(f"  Uniform Buffer sampling time: {uniform_time:.4f} seconds")
    print(f"  Regular Buffer sampling time: {regular_time:.4f} seconds")

def run_all_tests():
    """Run all buffer verification tests"""
    print("\n===== STARTING UNIFIED BUFFER VERIFICATION TESTS =====")
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Test standard buffer vs PER buffer
    test_per_buffer_regular_vs_uniform()
    
    # Test extreme priorities
    test_extreme_priorities()
    
    # Test n-step rewards computation
    test_n_step_rewards()
    
    # Test buffer performance
    test_buffer_performance()
    
    print("\n===== BUFFER VERIFICATION TESTS COMPLETED =====")

if __name__ == "__main__":
    run_all_tests() 