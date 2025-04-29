import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.append('.')

from src.agents.functions.buffers import PERBuffer

def test_per_buffer_regular_vs_uniform(save_folder):
    """Test PER buffer in both prioritized and uniform sampling modes"""
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
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(np.array(td_errors), bins=30, alpha=0.7, color='blue')
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
    ax4.hist(np.array(per_indices_all), bins=30, alpha=0.7, color='blue')
    ax4.set_title('PER Sampled Indices', fontsize=24)
    ax4.set_xlabel('Index', fontsize=20)
    ax4.set_ylabel('Frequency', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)
    plt.savefig(f'{save_folder}/weight_distribution.png')
    print(f"\nPlot saved to {save_folder}/weight_distribution.png")

def test_extreme_priorities(save_folder):
    """Test PER buffer with extreme priority differences to verify operation"""
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
    plt.savefig(f'{save_folder}/extreme_priority_test.png')
    print(f"Plot saved to {save_folder}/extreme_priority_test.png")

if __name__ == "__main__":
    print("Running PER buffer comprehensive tests...")
    test_per_buffer_regular_vs_uniform(save_folder='results/verification/buffer_tests')
    test_extreme_priorities(save_folder='results/verification/buffer_tests')
    print("\nAll tests completed!") 