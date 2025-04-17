### SAC ###
config_subsonic = {
    'hidden_dim_actor': 10,
    'number_of_hidden_layers_actor': 3,
    'hidden_dim_critic': 50,
    'number_of_hidden_layers_critic': 3,
    'temperature_initial': 0.01,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha_buffer': 0.6,
    'beta_buffer': 0.4,
    'beta_decay_buffer': 0.99,
    'buffer_size': 10000,
    'trajectory_length': 200,
    'batch_size': 64,
    'critic_learning_rate': 1e-7,
    'actor_learning_rate': 1e-7,
    'temperature_learning_rate': 1e-7,
    'critic_grad_max_norm': 1.0,
    'actor_grad_max_norm': 1.0,
    'temperature_grad_max_norm': 0.5,
    'max_std': 0.05,
    'num_episodes': 250,
    'critic_warm_up_steps': 1000,
    'pre_train_critic_learning_rate' : 1e-3,
    'pre_train_critic_batch_size' : 256
}

config_supersonic = {
    'hidden_dim_actor': 10,
    'number_of_hidden_layers_actor': 3,
    'hidden_dim_critic': 50,
    'number_of_hidden_layers_critic': 3,
    'temperature_initial': 0.01,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha_buffer': 0.6,
    'beta_buffer': 0.4,
    'beta_decay_buffer': 0.99,
    'buffer_size': 10000,
    'trajectory_length': 200,
    'batch_size': 64,
    'critic_learning_rate': 1e-7,
    'actor_learning_rate': 1e-7,
    'temperature_learning_rate': 1e-7,
    'critic_grad_max_norm': 1.0,
    'actor_grad_max_norm': 1.0,
    'temperature_grad_max_norm': 0.5,
    'max_std': 0.05,
    'num_episodes': 250,
    'critic_warm_up_steps': 1000,
    'pre_train_critic_learning_rate' : 1e-3,
    'pre_train_critic_batch_size' : 256
}

config_flip_over_boostbackburn = {
    'hidden_dim_actor': 10,
    'number_of_hidden_layers_actor': 3,
    'hidden_dim_critic': 50,
    'number_of_hidden_layers_critic': 3,
    'temperature_initial': 0.01,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha_buffer': 0.6,
    'beta_buffer': 0.4,
    'beta_decay_buffer': 0.99,
    'buffer_size': 10000,
    'trajectory_length': 200,
    'batch_size': 64,
    'critic_learning_rate': 1e-7,
    'actor_learning_rate': 1e-7,
    'temperature_learning_rate': 1e-7,
    'critic_grad_max_norm': 1.0,
    'actor_grad_max_norm': 1.0,
    'temperature_grad_max_norm': 0.5,
    'max_std': 0.05,
    'num_episodes': 250,
    'critic_warm_up_steps': 1000,
    'pre_train_critic_learning_rate' : 1e-3,
    'pre_train_critic_batch_size' : 256
}


### MARL ###

agent_config_marl = {
    'worker_agent' : config_subsonic,
    'central_agent' : config_subsonic,
    'number_of_workers': 2
}