### SAC ###

buffer_config_sac = {
    'alpha': 0.6,
    'beta': 0.4,
    'beta_decay': 0.99,
    'trajectory_length': 200,
    'buffer_size': 10000,   
}

agent_config_sac = {
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
    'critic_learning_rate': 1e-6,
    'actor_learning_rate': 1e-6,
    'temperature_learning_rate': 1e-6,
    'critic_grad_max_norm': 1.0,
    'actor_grad_max_norm': 1.0,
    'temperature_grad_max_norm': 0.01,
}


### MARL ###

agent_config_marl = {
    'worker_agent' : agent_config_sac,
    'central_agent' : agent_config_sac,
    'number_of_workers': 2
}