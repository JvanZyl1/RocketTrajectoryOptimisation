import time
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.envs.rl.env_wrapped_rl import rl_wrapped_env
from src.envs.rl.parallel_env_wrapped_rl import ParallelRocketEnv

def benchmark_sequential(env, num_steps: int = 1000):
    """Benchmark sequential environment steps"""
    print("Running sequential benchmark...")
    env.reset()
    
    start_time = time.time()
    for _ in tqdm(range(num_steps)):
        action = np.random.uniform(-1, 1, size=env.action_dim)
        env.step(action)
    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_second = num_steps / total_time
    return total_time, steps_per_second

def benchmark_parallel(parallel_env, num_steps: int = 1000):
    """Benchmark parallel environment steps"""
    print("Running parallel benchmark...")
    states = parallel_env.reset()
    
    start_time = time.time()
    for _ in tqdm(range(num_steps)):
        actions = np.random.uniform(-1, 1, size=(parallel_env.num_parallel_envs, parallel_env.action_dim))
        parallel_env.step(actions)
    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_second = (num_steps * parallel_env.num_parallel_envs) / total_time
    return total_time, steps_per_second

def run_benchmarks(flight_phase: str = 'subsonic', 
                  num_steps: int = 1000,
                  num_parallel_envs_list: list = [1, 2, 4, 6, 8]):
    """Run benchmarks for different numbers of parallel environments"""
    results = {
        'num_parallel_envs': [],
        'sequential_time': [],
        'parallel_time': [],
        'speedup': [],
        'efficiency': []
    }
    
    # Run sequential benchmark
    env = rl_wrapped_env(flight_phase=flight_phase)
    seq_time, seq_sps = benchmark_sequential(env, num_steps)
    env.close()
    
    # Run parallel benchmarks for different numbers of environments
    for num_envs in num_parallel_envs_list:
        print(f"\nTesting with {num_envs} parallel environments...")
        
        # Create parallel environment
        parallel_env = ParallelRocketEnv(
            flight_phase=flight_phase,
            num_parallel_envs=num_envs
        )
        
        # Run benchmark
        par_time, par_sps = benchmark_parallel(parallel_env, num_steps)
        parallel_env.close()
        
        # Calculate metrics
        speedup = seq_time / par_time
        efficiency = (speedup / num_envs) * 100  # Percentage of ideal speedup
        
        # Store results
        results['num_parallel_envs'].append(num_envs)
        results['sequential_time'].append(seq_time)
        results['parallel_time'].append(par_time)
        results['speedup'].append(speedup)
        results['efficiency'].append(efficiency)
        
        print(f"Results for {num_envs} parallel environments:")
        print(f"Sequential time: {seq_time:.2f}s")
        print(f"Parallel time: {par_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Efficiency: {efficiency:.1f}%")
    
    return results

def plot_results(results, save_path: str = 'results/parallel_benchmarks'):
    """Plot benchmark results"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot speedup
    ax1.plot(results['num_parallel_envs'], results['speedup'], 'o-', label='Actual Speedup')
    ax1.plot(results['num_parallel_envs'], results['num_parallel_envs'], 'k--', label='Ideal Speedup')
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('Speedup Factor')
    ax1.set_title('Speedup vs Number of Parallel Environments')
    ax1.grid(True)
    ax1.legend()
    
    # Plot efficiency
    ax2.plot(results['num_parallel_envs'], results['efficiency'], 'o-')
    ax2.axhline(y=100, color='k', linestyle='--', label='100% Efficiency')
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency vs Number of Environments')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/parallel_benchmark_results.png')
    plt.close()
    
    # Save results to file
    with open(f'{save_path}/benchmark_results.txt', 'w') as f:
        f.write("===== PARALLEL ENVIRONMENT BENCHMARK RESULTS =====\n")
        f.write(f"{'Num Envs':<10} {'Seq Time (s)':<15} {'Par Time (s)':<15} {'Speedup':<10} {'Efficiency (%)':<15}\n")
        f.write("-" * 65 + "\n")
        for i, n in enumerate(results['num_parallel_envs']):
            f.write(f"{n:<10} {results['sequential_time'][i]:<15.2f} {results['parallel_time'][i]:<15.2f} "
                   f"{results['speedup'][i]:<10.2f} {results['efficiency'][i]:<15.1f}\n")

if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmarks(
        flight_phase='subsonic',
        num_steps=1000,
        num_parallel_envs_list=[1, 2, 4, 6, 8]
    )
    
    # Plot and save results
    plot_results(results) 