import numpy as np
import jax.numpy as jnp
from src.agents.functions.buffer_utils import check_buffer_settings

def debug_buffer_weights(agent):
    """
    Debug function to check if PER weights are properly calculated.
    
    Args:
        agent: Agent containing the buffer to check
    """
    print("\n====== BUFFER WEIGHTS DEBUG ======")
    buffer = agent.buffer
    
    # Check if uniform sampling is enabled
    if hasattr(buffer, 'uniform_beun_fix_bool'):
        print(f"Current uniform_beun_fix_bool value: {buffer.uniform_beun_fix_bool}")
    else:
        print("Buffer does not have uniform_beun_fix_bool attribute!")
        return
    
    # Show detailed buffer info
    check_buffer_settings(buffer)
    
    # Try to sample with both settings
    orig_setting = buffer.uniform_beun_fix_bool
    
    # Force prioritized sampling
    print("\nTesting with prioritized sampling...")
    buffer.set_uniform_sampling(False)
    check_batch_weights(buffer)
    
    # Force uniform sampling
    print("\nTesting with uniform sampling...")
    buffer.set_uniform_sampling(True)
    check_batch_weights(buffer)
    
    # Restore original setting
    buffer.set_uniform_sampling(orig_setting)
    print(f"\nReverted to original setting: uniform_beun_fix_bool = {orig_setting}")
    
def check_batch_weights(buffer):
    """Sample a batch and print weight statistics"""
    import jax
    rng_key = jax.random.PRNGKey(0)
    _, _, _, _, _, _, weights = buffer(rng_key)
    
    # Convert to numpy for easier analysis
    weights_np = np.array(weights)
    
    print(f"Weight statistics:")
    print(f"  Min: {weights_np.min()}")
    print(f"  Max: {weights_np.max()}")
    print(f"  Mean: {weights_np.mean()}")
    print(f"  All equal to 1.0: {np.all(weights_np == 1.0)}")
    if not np.all(weights_np == 1.0):
        print(f"  Number of unique weights: {len(np.unique(weights_np))}")
        print(f"  First 5 weights: {weights_np[:5]}")
        
def toggle_buffer_uniform_sampling(agent):
    """Toggle between uniform and prioritized sampling"""
    buffer = agent.buffer
    if hasattr(buffer, 'uniform_beun_fix_bool'):
        # Toggle the setting
        new_value = not buffer.uniform_beun_fix_bool
        buffer.set_uniform_sampling(new_value)
        return new_value
    else:
        print("Buffer does not have uniform_beun_fix_bool attribute!")
        return None 