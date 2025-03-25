genetic_algorithm_params = {
    'population_size' : 80,
    'generations' : 200,
    'crossover_rate' : 0.75,
    'mutation_rate' : 0.25,
    'elite_size' : 10,
    'fitness_threshold' : -1000,
    'migration_interval' : 10,
    'num_migrants' : 8,
    'num_islands' : 6
}

pso_params = {
    'pop_size' : 1000,
    'generations' : 90,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 4,
    'communication_freq' : 10,                  # How often subswarms share information
    'migration_freq' : 5,                     # How often particles migrate
    'number_of_migrants' : 25,
    # Re-initialisation params
    're_initialise_number_of_particles' : 500,
    're_initialise_generation' : 45,
    # Non-heuristic optimiser params
    'local_search_solver' : 'particle_swarm_optimisation',
    'local_search_number_of_particles' : 1,
    'local_search_frequency' : 5,  
    'local_search_max_iter' : 200, # Null if particle swarm optimisation is used for the local search part.
    'local_search_trust_region_bounds_size' : 0.000001,
    'local': {
        'pop_size' : 20,
        'generations' : 25,
        'c1' : 1,
        'c2' : 1,
        'w_start' : 0.9,
        'w_end' : 0.4,
        'fitness_threshold' : -1000
    }
}