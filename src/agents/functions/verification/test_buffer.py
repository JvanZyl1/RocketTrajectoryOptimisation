import jax
import jax.numpy as jnp
import numpy as np
from src.agents.functions.buffers import PERBuffer
from src.agents.functions.debug_buffer import debug_buffer_weights, toggle_buffer_uniform_sampling
import argparse

def create_test_buffer(buffer_size=10000, batch_size=512):
    """Create a PER buffer with some random data for testing"""
    state_dim = 4
    action_dim = 2
    
    # Create buffer
    buffer = PERBuffer(
        gamma=0.99,
        alpha=0.6,
        beta=0.4,
        beta_decay=0.99,
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        trajectory_length=10,
        batch_size=batch_size
    )
    
    # Fill with random data with varying priorities
    rng_key = jax.random.PRNGKey(0)
    
    for i in range(buffer_size):
        # Create random data
        key1, key2, key3, key4, key5, rng_key = jax.random.split(rng_key, 6)
        state = jax.random.normal(key1, (state_dim,))
        action = jax.random.normal(key2, (action_dim,))
        reward = jax.random.normal(key3, ()) 
        next_state = jax.random.normal(key4, (state_dim,))
        done = False
        
        # Use varying TD errors - higher for later entries to make differences obvious
        td_error = jnp.array(i / buffer_size * 5.0 + jax.random.normal(key5, ()) * 0.1)
        
        buffer.add(state, action, reward, next_state, done, td_error)
    
    return buffer

def test_buffer_sampling():
    """Test prioritized sampling vs uniform sampling"""
    buffer = create_test_buffer()
    
    # Create a mock agent to use with debug functions
    class MockAgent:
        pass
    
    agent = MockAgent()
    agent.buffer = buffer
    
    # Run debug functions
    debug_buffer_weights(agent)
    
    # Test actual sampling differences
    print("\n===== Testing batch differences between uniform and prioritized sampling =====")
    
    # Sample with prioritized sampling 
    buffer.set_uniform_sampling(False)
    rng_key = jax.random.PRNGKey(42)
    _, _, _, _, _, indices_prioritized, _ = buffer(rng_key)
    
    # Sample with uniform sampling
    buffer.set_uniform_sampling(True)
    _, _, _, _, _, indices_uniform, _ = buffer(rng_key)
    
    # Convert to numpy arrays
    indices_prioritized = np.array(indices_prioritized)
    indices_uniform = np.array(indices_uniform)
    
    # Calculate overlap
    overlap = np.intersect1d(indices_prioritized, indices_uniform)
    
    print(f"Sampling comparison (using same random key):")
    print(f"  Same indices selected: {len(overlap)} out of {buffer.batch_size}")
    print(f"  Percentage overlap: {len(overlap) / buffer.batch_size * 100:.2f}%")
    
    # Check mean index values - should be higher for prioritized since later entries have higher priorities
    print(f"  Mean index (prioritized): {indices_prioritized.mean()}")
    print(f"  Mean index (uniform): {indices_uniform.mean()}")
    
    # Reset to prioritized
    buffer.set_uniform_sampling(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PER buffer functionality")
    parser.add_argument("--prioritized", action="store_true", help="Force prioritized sampling for tests")
    parser.add_argument("--uniform", action="store_true", help="Force uniform sampling for tests")
    
    args = parser.parse_args()
    
    test_buffer_sampling() 