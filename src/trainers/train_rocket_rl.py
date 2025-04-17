from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.critic_pre_train.pre_train_critic import pre_train_critic_from_pso_experiences
from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC as SAC_Trainer
from configs.agent_config import agent_config_sac

def train_rocket(number_of_episodes : int,
                 save_interval : int,
                 info : str = {},
                 load_network : bool = False,
                 critic_warm_up_steps : int = 0,
                 flight_phase : str = 'subsonic'):
    if load_network:
        if flight_phase == 'subsonic':
            model_name = 'subsonic_ascent'
        elif flight_phase == 'supersonic':
            model_name = 'supersonic_ascent'
        elif flight_phase == 'flip_over_boostbackburn':
            model_name = 'flip_over_boostbackburn'
        else:
            raise ValueError(f"Flight stage {flight_phase} not supported")
        
        actor_network, actor_params, hidden_dim, number_of_hidden_layers, state_dim, action_dim = load_pso_actor(model_name)
        agent_config_sac['hidden_dim_actor'] = hidden_dim
        agent_config_sac['number_of_hidden_layers_actor'] = number_of_hidden_layers

        critic_params_learner = pre_train_critic_from_pso_experiences(model_name = model_name,
                                                                        state_dim = state_dim,
                                                                        action_dim = action_dim,
                                                                        hidden_dim_critic = 50,
                                                                        number_of_hidden_layers_critic = 3,
                                                                        gamma = 0.99,
                                                                        tau = 0.005,
                                                                        critic_learning_rate = 1e-3,
                                                                        batch_size = 32)
        critic_params, critic_target_params, critic_opt_state = critic_params_learner()

        
        
        trainer = SAC_Trainer(agent_config = agent_config_sac,
                              number_of_episodes = number_of_episodes,
                              save_interval = save_interval,
                              info = info,
                              actor_params = actor_params,
                              critic_params = critic_params,
                              critic_target_params = critic_target_params,
                              critic_opt_state = critic_opt_state,
                              critic_warm_up_steps = critic_warm_up_steps,
                              experiences_model_name = model_name,
                              flight_phase = flight_phase)
    else:
        trainer = SAC_Trainer(agent_config = agent_config_sac,
                                number_of_episodes = number_of_episodes,
                                save_interval = save_interval,
                                info = info,
                                critic_warm_up_steps = critic_warm_up_steps,
                                flight_phase = flight_phase)
    trainer.train()

