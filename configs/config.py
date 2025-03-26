pso_params = {
    'pop_size' : 5000,
    'generations' : 90,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 4,
    'communication_freq' : 10,                  # How often subswarms share information
    'migration_freq' : 5,                     # How often particles migrate
    'number_of_migrants' : 5,
    # Re-initialisation params
    're_initialise_number_of_particles' : 4000,
    're_initialise_generation' : 45,
}