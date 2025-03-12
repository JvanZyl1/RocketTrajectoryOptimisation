from net_load import create_and_load_network

def train_rocket(agent_type : str, # 'SAC', 'MARL', 'MARL_CTDE'
                 number_of_episodes : int,
                 save_interval : int,
                 info : str = {},
                 marl_load_info : str = None):
    if agent_type == 'SAC':
        from src.trainers.trainer_rocket_SAC import VerticalRisingTrain as SAC_Trainer
        from configs.agent_config import agent_config_sac

        actor_network, actor_params, hidden_dim, number_of_hidden_layers = create_and_load_network()

        agent_config_sac['hidden_dim_actor'] = hidden_dim
        agent_config_sac['number_of_hidden_layers_actor'] = number_of_hidden_layers
        trainer = SAC_Trainer(agent_config = agent_config_sac,
                            number_of_episodes = number_of_episodes,
                            save_interval = save_interval,
                            info = info,
                            actor_params = actor_params,
                            tqdm_bool = True,
                            print_bool = False)
    
    elif agent_type == 'MARL':
        from src.trainers.trainer_rocket_MARL import VerticalRisingTrain as MARL_Trainer
        from configs.agent_config import agent_config_marl
        trainer = MARL_Trainer(num_episodes = number_of_episodes,
                            worker_agent_config = agent_config_marl['worker_agent'],
                            central_agent_config = agent_config_marl['central_agent'],
                            print_bool = False,
                            tqdm_bool = True,
                            save_interval = save_interval,
                            number_of_agents = agent_config_marl['number_of_workers'],
                            info = info,
                            marl_load_info = marl_load_info)
    
    elif agent_type == 'MARL_CTDE':
        from src.trainers.trainer_rocket_MARL_CTDE import VerticalRisingTrain as MARL_CTDE_Trainer
        from configs.agent_config import agent_config_marl_ctde
        trainer = MARL_CTDE_Trainer(number_of_episodes = number_of_episodes,
                                    agent_config = agent_config_marl_ctde,
                                    info = info,
                                    save_interval = save_interval,
                                    tqdm_bool = True,
                                    print_bool = False)
    else:
        raise ValueError(f"Agent type {agent_type} not supported")
    
    trainer.train()
