genetic_algorithm_params = {
    'population_size' : 25,
    'generations' : 5,
    'crossover_rate' : 0.75,
    'mutation_rate' : 0.05,
    'fitness_threshold' : -150,
    'migration_interval' : 10,
    'num_migrants' : 8,
    'num_islands' : 5
}

pso_params = {
    'pop_size' : 50,
    'generations' : 5,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -150,
    'num_sub_swarms' : 8
}