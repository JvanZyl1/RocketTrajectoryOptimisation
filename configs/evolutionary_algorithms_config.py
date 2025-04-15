subsonic_pso_params = {
    'pop_size' : 1000,
    'generations' : 300,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.7,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 2,
    'communication_freq' : 10,                  # How often subswarms share information
    'migration_freq' : 5,                     # How often particles migrate
    'number_of_migrants' : 15,
    # Re-initialisation params
    're_initialise_number_of_particles' : 600,
    're_initialise_generation' : 90,
}

supersonic_pso_params = {
    'pop_size' : 700,
    'generations' : 300,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 3,
    'communication_freq' : 10,                  # How often subswarms share information
    'migration_freq' : 5,                     # How often particles migrate
    'number_of_migrants' : 15,
    # Re-initialisation params
    're_initialise_number_of_particles' : 1000,
    're_initialise_generation' : 250,
}

flip_over_pso_params = {
    'pop_size' : 700,
    'generations' : 300,
    'c1' : 1,
    'c2' : 1,
    'w_start' : 0.9,
    'w_end' : 0.4,
    'fitness_threshold' : -1000,
    'num_sub_swarms' : 3,
    'communication_freq' : 10,                  # How often subswarms share information
    'migration_freq' : 5,                     # How often particles migrate
    'number_of_migrants' : 15,
    # Re-initialisation params
    're_initialise_number_of_particles' : 1000,
    're_initialise_generation' : 250,
}