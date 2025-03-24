genetic_algorithm_params = {
    'population_size' : 80,
    'generations' : 200,
    'crossover_rate' : 0.75,
    'mutation_rate' : 0.05,
    'elite_size' : 10,
    'fitness_threshold' : -100,
    'migration_interval' : 10,
    'num_migrants' : 8,
    'num_islands' : 2
}

pso_params = {
    'pop_size' : 100,
    'generations' : 50,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 3,
    # Non-heuristic optimiser params
    'local_search_solver' : 'trust-constr',
    'local_search_number_of_particles' : 2,
    'local_search_frequency' : 10,
    'local_search_max_iter' : 200
}