from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.sac_marl_ctde import SAC_MARL_CTDE
import pickle

def load_sac(agent_path):
    # Read pickle file
    with open(agent_path, 'rb') as f:
        agent_config = pickle.load(f)

    # Agent initalisations : all numpy atm.
    seed = agent_config['seed']
    state_dim = agent_config['dimensions']['state_dim']
    action_dim = agent_config['dimensions']['action_dim']
    hidden_dim_actor = agent_config['dimensions']['hidden_dim_actor']
    hidden_dim_critic = agent_config['dimensions']['hidden_dim_critic']
    std_min = agent_config['dimensions']['std_min']
    std_max = agent_config['dimensions']['std_max']
    gamma = agent_config['hyperparameters']['gamma']
    tau = agent_config['hyperparameters']['tau']
    temperature_initial = agent_config['hyperparameters']['temperature_initial']
    critic_grad_max_norm = agent_config['hyperparameters']['critic_grad_max_norm']
    critic_lr = agent_config['hyperparameters']['critic_lr']
    critic_weight_decay = agent_config['hyperparameters']['critic_weight_decay']
    actor_grad_max_norm = agent_config['hyperparameters']['actor_grad_max_norm']
    actor_lr = agent_config['hyperparameters']['actor_lr']
    actor_weight_decay = agent_config['hyperparameters']['actor_weight_decay']
    temperature_lr = agent_config['hyperparameters']['temperature_lr']
    temperature_grad_max_norm = agent_config['hyperparameters']['temperature_grad_max_norm']
    alpha_buffer = agent_config['buffer_state']['static_config']['alpha']
    beta_buffer = agent_config['buffer_state']['beta']
    beta_decay_buffer = agent_config['buffer_state']['static_config']['beta_decay']
    trajectory_length = agent_config['buffer_state']['trajectory_length']
    buffer_size = agent_config['buffer_state']['static_config']['buffer_size']
    save_path = agent_config['save_path']
    print_bool = agent_config['print_bool']

    # Agent initialisation
    agent = Agent(seed=seed,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  hidden_dim_actor=hidden_dim_actor,
                  hidden_dim_critic=hidden_dim_critic,
                  std_min=std_min,
                  std_max=std_max,
                  gamma=gamma,
                  tau=tau,
                  temperature_initial=temperature_initial,
                  critic_grad_max_norm=critic_grad_max_norm,
                  critic_lr=critic_lr,
                  critic_weight_decay=critic_weight_decay,
                  actor_grad_max_norm=actor_grad_max_norm,
                  actor_lr=actor_lr,
                  actor_weight_decay=actor_weight_decay,
                  temperature_lr=temperature_lr,
                  temperature_grad_max_norm=temperature_grad_max_norm,
                  alpha_buffer=alpha_buffer,
                  beta_buffer=beta_buffer,
                  beta_decay_buffer=beta_decay_buffer,
                  trajectory_length=trajectory_length,
                  buffer_size=buffer_size,
                  save_path=save_path,
                  print_bool=print_bool)
    
    # Update buffer
    agent.buffer.priorities = agent_config['buffer_state']['priorities']
    agent.buffer.buffer = agent_config['buffer_state']['buffer']
    agent.buffer.position = agent_config['buffer_state']['position']
    agent.buffer.n_step_buffer = agent_config['buffer_state']['n_step_buffer']

    # Update agent
    agent.target_entropy = agent_config['target_entropy']
    agent.rng_key = agent_config['rng_key']

    agent.actor_params = agent_config['actor_params']
    agent.critic_params = agent_config['critic_params']
    agent.temperature = agent_config['temperature']

    agent.actor_opt_state = agent_config['actor_opt_state']
    agent.critic_opt_state = agent_config['critic_opt_state']
    agent.temperature_opt_state = agent_config['temperature_opt_state']

    agent.critic_target_params = agent_config['critic_target_params']

    return agent

marl_agent_config = {

    'worker_agent' : {
        'worker_actor_lr': 3e-3,
        'worker_temperature_lr': 5e-4,
        'actor_grad_max_norm_worker': 1.0,
        'temperature_grad_max_norm_worker': 1.0,
        'worker_temperature': 0.1
    },
    'number_of_workers': 2
}

def load_sac_marl_ctde(agent_path):
    # Read pickle file
    with open(agent_path, 'rb') as f:
        agent_config = pickle.load(f)

    # Agent initalisations : all numpy atm.
    seed = agent_config['seed']
    state_dim = agent_config['dimensions']['state_dim']
    action_dim = agent_config['dimensions']['action_dim']
    save_path = agent_config['save_path']
    print_bool = agent_config['print_bool']

    config = {}
    config['gamma'] = agent_config['gamma']
    buffer_config = {
        'alpha': agent_config['buffer_params']['alpha'],
        'beta': agent_config['buffer_params']['beta'],
        'beta_decay': agent_config['buffer_params']['beta_decay'],
        'buffer_size': agent_config['buffer_params']['buffer_size'],
        'batch_size': agent_config['buffer_params']['batch_size'],
        'trajectory_length': agent_config['buffer_params']['trajectory_length'],
    }
    config['buffer'] = buffer_config
    config['batch_size'] = agent_config['batch_size']
    config['hidden_dim_actor'] = agent_config['hidden_dim_actor']
    config['hidden_dim_critic'] = agent_config['hidden_dim_critic']
    config['std_min'] = agent_config['std_min']
    config['std_max'] = agent_config['std_max']
    central_agent_config = {
        'central_actor_lr': agent_config['central_agent']['central_actor_lr'],
        'central_critic_lr': agent_config['central_agent']['central_critic_lr'],
        'central_temperature_lr': agent_config['central_agent']['central_temperature_lr'],
        'critic_grad_max_norm_central': agent_config['central_agent']['critic_grad_max_norm_central'],
        'temperature_grad_max_norm_central': agent_config['central_agent']['temperature_grad_max_norm_central'],
        'actor_grad_max_norm_central': agent_config['central_agent']['actor_grad_max_norm_central'],
        'central_temperature': agent_config['central_agent_build']['central_temperature']
    }
    config['central_agent'] = central_agent_config
    worker_agents_config = {
        'worker_actor_lr': agent_config['worker_agent']['worker_actor_lr'],
        'worker_temperature_lr': agent_config['worker_agent']['worker_temperature_lr'],
        'actor_grad_max_norm_worker': agent_config['worker_agent']['actor_grad_max_norm_worker'],
        'temperature_grad_max_norm_worker': agent_config['worker_agent']['temperature_grad_max_norm_worker'],
        'worker_temperature': 0.1 # dummy value
    }
    config['worker_agent'] = worker_agents_config
    config['number_of_workers'] = agent_config['number_of_workers']

    # Create agent
    agent = SAC_MARL_CTDE(seed=seed,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          config=agent_config,
                          save_path=save_path,
                          print_bool=print_bool)
    
    # Update buffer
    agent.buffer.priorities = agent_config['buffer_params']['priorities']
    agent.buffer.buffer = agent_config['buffer_params']['buffer']
    agent.buffer.position = agent_config['buffer_params']['position']
    agent.buffer.n_step_buffer = agent_config['buffer_params']['n_step_buffer']

    # Update agent
    agent.target_entropy = agent_config['target_entropy']
    agent.rng_key = agent_config['rng_key']

    # Update central agent
    agent.central_actor_params = agent_config['central_agent_build']['central_actor_params']
    agent.central_critic_params = agent_config['central_agent_build']['central_critic_params']
    agent.central_critic_target_params = agent_config['central_agent_build']['central_critic_target_params']
    agent.central_actor_opt_state = agent_config['central_agent_build']['central_actor_opt_state']
    agent.central_critic_opt_state = agent_config['central_agent_build']['central_critic_opt_state']
    agent.central_temperature_opt_state = agent_config['central_agent_build']['central_temperature_opt_state']
    agent.central_temperature = agent_config['central_agent_build']['central_temperature']
    
    # Update worker agents
    agent.all_worker_actor_params = agent_config['worker_agent_build']['all_worker_actor_params']
    agent.all_worker_actor_opt_state = agent_config['worker_agent_build']['all_worker_actor_opt_state']
    agent.all_worker_temperatures = agent_config['worker_agent_build']['all_worker_temperatures']
    agent.all_worker_temperature_opt_state = agent_config['worker_agent_build']['all_worker_temperature_opt_state']
    
    return agent