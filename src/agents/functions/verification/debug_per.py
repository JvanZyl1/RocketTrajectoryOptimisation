import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC

def debug_per_buffer(trainer_instance):
    """Debug the PER buffer priorities and sampling weights"""
    agent = trainer_instance.agent
    buffer = agent.buffer
    
    # Check and report status
    print(f"PER Status:")
    print(f"  Using uniform sampling: {buffer.is_using_uniform_sampling()}")
    print(f"  Buffer size: {buffer.buffer_size}")
    print(f"  Current position: {buffer.position}")
    print(f"  Alpha (priority exponent): {buffer.alpha}")
    print(f"  Beta (importance sampling): {buffer.beta}")
    
    # Analyze priorities
    if buffer.position > 0:
        # Only look at filled portion of buffer
        priorities = np.array(buffer.priorities[:buffer.position])
        
        print(f"\nPriorities Statistics:")
        print(f"  Min: {np.min(priorities)}")
        print(f"  Max: {np.max(priorities)}")
        print(f"  Mean: {np.mean(priorities)}")
        print(f"  Std: {np.std(priorities)}")
        print(f"  All equal: {np.allclose(priorities, priorities[0], rtol=1e-5)}")
        
        # Count elements very close to the minimum value
        close_to_min = np.isclose(priorities, np.min(priorities), rtol=1e-5)
        print(f"  Number of priorities at minimum: {np.sum(close_to_min)} out of {len(priorities)}")
        
        # Plot histogram of priorities
        plt.figure(figsize=(10, 6))
        plt.hist(priorities, bins=50)
        plt.title("Distribution of Priorities")
        plt.xlabel("Priority Value")
        plt.ylabel("Count")
        plt.savefig("priorities_histogram.png")
        plt.close()
    
    # Test sampling and weight calculation
    print("\nSampling Test:")
    
    # Force prioritized sampling and verify
    print("Testing with prioritized sampling...")
    buffer.set_uniform_sampling(False)
    
    # Sample from buffer
    rng_key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
    for i in range(5):  # Try multiple samples
        states, actions, rewards, next_states, dones, indices, weights = buffer(rng_key)
        rng_key, _ = jax.random.split(rng_key)
        
        weights_np = np.array(weights)
        indices_np = np.array(indices)
        
        print(f"\nSample {i+1}:")
        print(f"  Weight statistics:")
        print(f"    Min: {np.min(weights_np)}")
        print(f"    Max: {np.max(weights_np)}")
        print(f"    Mean: {np.mean(weights_np)}")
        print(f"    Std: {np.std(weights_np)}")
        print(f"    All equal to 1.0: {np.all(weights_np == 1.0)}")
        print(f"    Unique weight values: {len(np.unique(weights_np))}")
        
        # Calculate what weights should be based on priorities
        if buffer.position > 0:
            # Get sampled priorities
            sampled_priorities = np.array(buffer.priorities[indices_np])
            
            # Sum for normalization
            sum_priorities_alpha = np.sum(np.array(buffer.priorities[:buffer.position]) ** buffer.alpha)
            
            # Calculate probabilities and expected weights
            if sum_priorities_alpha > 0:
                probs = (sampled_priorities ** buffer.alpha) / sum_priorities_alpha
                expected_weights = (probs * buffer.position) ** (-buffer.beta)
                expected_weights = expected_weights / np.max(expected_weights)
                
                print(f"  Expected weights (manually calculated):")
                print(f"    Min: {np.min(expected_weights)}")
                print(f"    Max: {np.max(expected_weights)}")
                print(f"    Mean: {np.mean(expected_weights)}")
                print(f"    Std: {np.std(expected_weights)}")
                print(f"    Matches actual: {np.allclose(weights_np, expected_weights, rtol=1e-5)}")
    
    # Now verify weight calculation by forcing extreme priorities
    print("\nTesting with extreme priority differences:")
    
    # Create a modified priorities array for testing
    test_priorities = np.ones_like(buffer.priorities)
    if buffer.position > 100:
        # Set first 50 elements to minimum value
        test_priorities = test_priorities.at[:50].set(1e-6)
        # Set next 50 elements to high values
        test_priorities = test_priorities.at[50:100].set(100.0)
        
        # Save original priorities
        original_priorities = buffer.priorities
        
        # Set test priorities
        buffer.priorities = jnp.array(test_priorities)
        
        # Sample again
        states, actions, rewards, next_states, dones, indices, weights = buffer(rng_key)
        weights_np = np.array(weights)
        
        print(f"  With extreme priorities:")
        print(f"    Min weight: {np.min(weights_np)}")
        print(f"    Max weight: {np.max(weights_np)}")
        print(f"    All equal to 1.0: {np.all(weights_np == 1.0)}")
        
        # Restore original priorities
        buffer.priorities = original_priorities
    
    return "Debug complete. Check the console output and priorities_histogram.png"

if __name__ == "__main__":
    # Create trainer but don't start training yet
    trainer = RocketTrainer_SAC(flight_phase='subsonic',
                              load_from='supervisory',
                              load_buffer_bool=True,  # We want to debug the filled buffer
                              save_interval=5,
                              pre_train_critic_bool=False,
                              enable_wind=True)
    
    # Run the debug function
    result = debug_per_buffer(trainer)
    print(result) 