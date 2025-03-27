from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.pre_train_critic import pre_train_critic_from_pso_experiences
from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC as SAC_Trainer
from configs.agent_config import agent_config_sac

from src.trainers.trainer_rocket_MARL import VerticalRisingTrain as MARL_Trainer
from configs.agent_config import agent_config_marl

from src.envs.rl.env_wrapped_StableBaselines3 import compile_StableBaselines3_env
from src.trainers.trainers import trainer_StableBaselines3

def train_rocket(agent_type : str, # 'SAC', 'MARL', 'StableBaselines3'
                 number_of_episodes : int,
                 save_interval : int,
                 info : str = {},
                 load_network : bool = False,
                 marl_load_info : str = None,
                 critic_warm_up_steps : int = 0):
    if agent_type == 'SAC':
        if load_network:
            actor_network, actor_params, hidden_dim, number_of_hidden_layers = load_pso_actor()
            agent_config_sac['hidden_dim_actor'] = hidden_dim
            agent_config_sac['number_of_hidden_layers_actor'] = number_of_hidden_layers

            critic_params_learner = pre_train_critic_from_pso_experiences()
            critic_params, critic_target_params, critic_opt_state = critic_params_learner()
            
            trainer = SAC_Trainer(agent_config = agent_config_sac,
                                  number_of_episodes = number_of_episodes,
                                  save_interval = save_interval,
                                  info = info,
                                  actor_params = actor_params,
                                  critic_params = critic_params,
                                  critic_target_params = critic_target_params,
                                  critic_opt_state = critic_opt_state,
                                  critic_warm_up_steps = critic_warm_up_steps)
        else:
            trainer = SAC_Trainer(agent_config = agent_config_sac,
                                  number_of_episodes = number_of_episodes,
                                  save_interval = save_interval,
                                  info = info,
                                  critic_warm_up_steps = critic_warm_up_steps)
        trainer.train()
    
    elif agent_type == 'MARL':
        trainer = MARL_Trainer(num_episodes = number_of_episodes,
                            worker_agent_config = agent_config_marl['worker_agent'],
                            central_agent_config = agent_config_marl['central_agent'],
                            save_interval = save_interval,
                            number_of_agents = agent_config_marl['number_of_workers'],
                            info = info,
                            marl_load_info = marl_load_info)
        trainer.train()
    
    elif agent_type == 'StableBaselines3':
        env = compile_StableBaselines3_env(model_name = 'sac_endo_ascent',
                                        norm_obs = True,
                                        norm_reward = False)

        trainer_StableBaselines3(env,
                                model_name = 'sac_endo_ascent')
    else:
        raise ValueError(f"Agent type {agent_type} not supported")
    
    
