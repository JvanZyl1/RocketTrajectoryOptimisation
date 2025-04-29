import numpy as np

def check_buffer_settings(buffer):
    """
    Print diagnostic information about a buffer's settings and state
    
    Args:
        buffer: A PERBuffer instance to check
    """
    print("\n====== Buffer Diagnostics ======")
    print(f"Using uniform sampling: {buffer.is_using_uniform_sampling()}")
    print(f"Buffer size: {buffer.buffer_size}")
    print(f"Current position: {buffer.position}")
    print(f"Batch size: {buffer.batch_size}")
    print(f"Alpha (priority exponent): {buffer.alpha}")
    print(f"Beta (importance sampling): {buffer.beta}")
    print(f"Beta decay: {buffer.beta_decay}")
    
    # Check min, max, mean priorities
    if buffer.position > 0:
        filled_priorities = buffer.priorities[:buffer.position]
        print(f"Priority stats (filled entries):")
        print(f"  Min: {np.min(np.array(filled_priorities))}")
        print(f"  Max: {np.max(np.array(filled_priorities))}")
        print(f"  Mean: {np.mean(np.array(filled_priorities))}")
        print(f"  Std: {np.std(np.array(filled_priorities))}")
    
    # Sample a batch and check weights
    import jax
    rng_key = jax.random.PRNGKey(0)
    states, actions, rewards, next_states, dones, indices, weights = buffer(rng_key)
    
    print("\nSampled batch weights:")
    print(f"  Min: {np.min(np.array(weights))}")
    print(f"  Max: {np.max(np.array(weights))}")
    print(f"  Mean: {np.mean(np.array(weights))}")
    print(f"  Std: {np.std(np.array(weights))}")
    print(f"  All equal to 1.0: {np.all(np.array(weights) == 1.0)}")
    print("==============================\n") 