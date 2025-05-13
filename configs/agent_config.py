config_subsonic = {
    'sac' : {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 250,
        'number_of_hidden_layers_critic': 4,  # increased to 10 for next run.
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 30000, # 50000 in buffer atm
        'trajectory_length': 200,
        'batch_size': 512,
        'critic_learning_rate': 1e-7,
        'actor_learning_rate': 1e-7,
        'temperature_learning_rate': 6e-7,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.8,
        'max_std': 0.2,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 256,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 256,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 1e-3, # Also for critic warm-up
        'actor_learning_rate': 3e-5, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.2/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.2,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.008,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-4, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}

config_supersonic = {
    'sac' : {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 250,
        'number_of_hidden_layers_critic': 4,  # increased to 10 for next run.
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 30000, # 50000 in buffer atm
        'trajectory_length': 200,
        'batch_size': 512,
        'critic_learning_rate': 1e-7,
        'actor_learning_rate': 1e-7,
        'temperature_learning_rate': 6e-7,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.8,
        'max_std': 0.01,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 256,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 256,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 1e-3, # Also for critic warm-up
        'actor_learning_rate': 3e-5, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.2/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.2,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.006,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-4, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}


config_flip_over_boostbackburn = {
    'sac' : {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 250,
        'number_of_hidden_layers_critic': 4,  # increased to 10 for next run.
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 30000, # 50000 in buffer atm
        'trajectory_length': 200,
        'batch_size': 512,
        'critic_learning_rate': 1e-7,
        'actor_learning_rate': 1e-7,
        'temperature_learning_rate': 6e-7,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.8,
        'max_std': 0.01,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 256,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 256,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 1e-3, # Also for critic warm-up
        'actor_learning_rate': 3e-5, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.2/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.2,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.006,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-4, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}

config_ballistic_arc_descent = {
    'sac' : {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 250,
        'number_of_hidden_layers_critic': 4,  # increased to 10 for next run.
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 30000, # 50000 in buffer atm
        'trajectory_length': 200,
        'batch_size': 512,
        'critic_learning_rate': 1e-7,
        'actor_learning_rate': 1e-7,
        'temperature_learning_rate': 6e-7,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.8,
        'max_std': 0.01,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 256,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 230,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 20000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 1e-3, # Also for critic warm-up
        'actor_learning_rate': 3e-5, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.2/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.2,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.008,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-5, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}
config_re_entry_burn = {
    'sac' : {
        'hidden_dim_actor': 50,
        'number_of_hidden_layers_actor': 14,
        'hidden_dim_critic': 250,
        'number_of_hidden_layers_critic': 4,  # increased to 10 for next run.
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 50000 in buffer atm
        'trajectory_length': 200,
        'batch_size': 512,
        'critic_learning_rate': 1e-7,
        'actor_learning_rate': 1e-7,
        'temperature_learning_rate': 6e-7,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 0.8,
        'max_std': 0.01,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 256,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 230,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 4e-4, # Also for critic warm-up
        'actor_learning_rate': 3e-3, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.4/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.4,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.0085,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-5, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}


config_landing_burn = {
    'sac' : {
        'hidden_dim_actor': 128,
        'number_of_hidden_layers_actor': 3,
        'hidden_dim_critic': 512,
        'number_of_hidden_layers_critic': 8,
        'temperature_initial': 0.1,
        'gamma': 0.85,
        'tau': 0.01,
        'alpha_buffer': 0.4,
        'beta_buffer': 0.6,
        'beta_decay_buffer': 0.99,
        'buffer_size': 20000, # 50000 in buffer atm
        'trajectory_length': 10,
        'batch_size': 256,
        'critic_learning_rate': 7e-3,
        'actor_learning_rate': 1e-3,
        'temperature_learning_rate': 6e-3,
        'critic_grad_max_norm': 0.2,
        'actor_grad_max_norm': 0.2,
        'temperature_grad_max_norm': 1.0,
        'max_std': 1.0,
        'l2_reg_coef': 0.001,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 230,
        'number_of_hidden_layers_actor': 5,
        'hidden_dim_critic': 230,
        'number_of_hidden_layers_critic': 5,
        'gamma': 0.99,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 10000, # 25000 -> 50000
        'trajectory_length': 7,
        'batch_size': 512,
        'critic_learning_rate': 4e-3, # Also for critic warm-up
        'actor_learning_rate': 3e-3, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 0.5,
        'actor_grad_max_norm': 0.5,
        'policy_noise': 0.1/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.1,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.003,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 1650,
    'critic_warm_up_steps': 2000,
    'pre_train_critic_learning_rate' : 1e-5, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 6,
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
}