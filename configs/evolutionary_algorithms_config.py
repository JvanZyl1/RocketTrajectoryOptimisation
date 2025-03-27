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
    'pop_size' : 5000,
    'generations' : 85,
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
    're_initialise_number_of_particles' : 1000,
    're_initialise_generation' : 20,

}