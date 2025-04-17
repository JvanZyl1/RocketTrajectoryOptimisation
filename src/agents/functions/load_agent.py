import pickle

from src.agents.soft_actor_critic import SoftActorCritic
from torch.utils.tensorboard import SummaryWriter

def load_sac(agent_path : str):
    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    sac = SoftActorCritic(**agent_state['inputs'])

    sac.rng_key = agent_state['misc']['rng_key']
    sac.run_id = agent_state['misc']['run_id']
    sac.writer = SummaryWriter(log_dir=f'data/agent_saves/VanillaSAC/{sac.flight_phase}/runs/{sac.run_id}')


    # Update logging all
    sac.critic_loss_episode = agent_state['logging']['critic_loss_episode']
    sac.actor_loss_episode = agent_state['logging']['actor_loss_episode']
    sac.temperature_loss_episode = agent_state['logging']['temperature_loss_episode']
    sac.td_errors_episode = agent_state['logging']['td_errors_episode']
    sac.temperature_values_all_episode = agent_state['logging']['temperature_values_all_episode']
    sac.number_of_steps_episode = agent_state['logging']['number_of_steps_episode']
    sac.critic_losses = agent_state['logging']['critic_losses']
    sac.actor_losses = agent_state['logging']['actor_losses']
    sac.temperature_losses = agent_state['logging']['temperature_losses']
    sac.td_errors = agent_state['logging']['td_errors']
    sac.temperature_values = agent_state['logging']['temperature_values']
    sac.number_of_steps = agent_state['logging']['number_of_steps']
    sac.episode_idx = agent_state['logging']['episode_idx']
    sac.step_idx = agent_state['logging']['step_idx']

    # Update update all
    sac.critic_params = agent_state['update']['critic_params']
    sac.critic_opt_state = agent_state['update']['critic_opt_state']
    sac.critic_target_params = agent_state['update']['critic_target_params']
    sac.actor_params = agent_state['update']['actor_params']
    sac.actor_opt_state = agent_state['update']['actor_opt_state']
    sac.temperature = agent_state['update']['temperature']
    sac.temperature_opt_state = agent_state['update']['temperature_opt_state']
    sac.buffer = agent_state['update']['buffer']  
    
    return sac