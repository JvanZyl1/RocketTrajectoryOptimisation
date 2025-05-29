from src.particle_swarm_optimisation.parallel_pso import ParallelParticleSubswarmOptimisation
import multiprocessing as mp

if __name__ == "__main__":
    # Get number of available CPU cores
    num_cpus = mp.cpu_count()
    print(f"System has {num_cpus} CPU cores available")
    
    # Choose flight phase
    flight_phase = 'landing_burn_pure_throttle'  # 'subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent'
    
    # Initialize the parallel PSO with 80% of available cores (leave some for system)
    num_processes = max(1, int(num_cpus * 0.8))
    
    print(f"Running parallel PSO optimization for {flight_phase} using {num_processes} processes")
    
    # Create and run the parallel PSO optimizer
    parallel_pso = ParallelParticleSubswarmOptimisation(
        flight_phase=flight_phase,
        save_interval=1,
        enable_wind=False,
        num_processes=num_processes
    )
    
    # Run the optimization
    best_position, best_fitness = parallel_pso()
    
    print(f"Optimization completed!")
    print(f"Best fitness: {best_fitness}")
    print(f"Best position: {best_position}") 