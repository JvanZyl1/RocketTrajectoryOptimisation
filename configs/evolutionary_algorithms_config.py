genetic_algorithm_params = {
    'population_size' : 400,
    'generations' : 250,
    'crossover_rate' : 0.75,
    'mutation_rate' : 0.25,
    'elite_size' : 10,
    'fitness_threshold' : -1000,
    'migration_interval' : 10,
    'num_migrants' : 8,
    'num_islands' : 6
}

pso_params = {
    'pop_size' : 5000,                                   # Number of particles
    'generations' : 2000,                                # Number of generations
    'c1' : 1,                                            # Cognitive parameter; how much the particle moves towards its personal best
    'c2' : 1,                                            # Social parameter; how much the particle moves towards the global best
    'w_start' : 0.8,                                     # Initial inertia weight; how much the particle moves towards the previous velocity
    'w_end' : 0.4,                                       # Final inertia weight; how much the particle moves towards the previous velocity
    'fitness_threshold' : -1000,                         # Fitness threshold; the fitness value below which the algorithm stops
    'num_sub_swarms' : 2                                 # Number of sub-swarms; the number of sub-populations in the PSO
}