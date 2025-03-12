### SAC ###

buffer_config_sac = {
    'alpha': 0.6,
    'beta': 0.4,
    'beta_decay': 0.99,
    'trajectory_length': 100,
    'buffer_size': 10000,   
}

agent_config_sac = {
    'hidden_dim_actor': 30,
    'hidden_dim_critic': 30,
    'std_min': 0.001,
    'std_max': 0.05,
    'gamma': 0.95,
    'tau': 0.1,
    'temperature_initial': 0.1,
    'critic_grad_max_norm': 1.0,
    'critic_lr': 1e-3,
    'critic_weight_decay': 0.0,
    'actor_grad_max_norm': 0.0001,
    'actor_lr': 4e-5,
    'actor_weight_decay': 0.0,
    'temperature_lr': 1e-25,
    'temperature_grad_max_norm': 1.0,
    'alpha_buffer': buffer_config_sac['alpha'],
    'beta_buffer': buffer_config_sac['beta'],
    'beta_decay_buffer': buffer_config_sac['beta_decay'],
    'trajectory_length': buffer_config_sac['trajectory_length'],
    'buffer_size': buffer_config_sac['buffer_size'],
    'batch_size': 64
}


### MARL ###

agent_config_marl = {
    'worker_agent' : agent_config_sac,
    'central_agent' : agent_config_sac,
    'number_of_workers': 2
}

### MARL_CTDE ###

agent_config_marl_ctde = {
    'gamma': 0.99,
    'tau': 0.005,
    'buffer' : {
        'alpha': 0.6,
        'beta': 0.4,
        'beta_decay': 0.99,
        'buffer_size': 100000,
        'batch_size': 128,
        'trajectory_length': 5,
    },
    'batch_size': 128,
    'hidden_dim_actor': 56,
    'hidden_dim_critic': 56,
    'std_min': 1e6,
    'std_max': 1e9,
    'central_agent' : {
        'central_actor_lr': 3e-6,
        'central_critic_lr': 3e-6,
        'central_temperature_lr': 5e-6,
        'critic_grad_max_norm_central': 1.0,
        'temperature_grad_max_norm_central': 1.0,
        'actor_grad_max_norm_central': 10.0,
        'central_temperature': 0.1
    },
    'worker_agent' : {
        'worker_actor_lr': 3e-3,
        'worker_temperature_lr': 5e-3,
        'actor_grad_max_norm_worker': 1.0,
        'temperature_grad_max_norm_worker': 1.0,
        'worker_temperature': 0.1
    },
    'number_of_workers': 2
}