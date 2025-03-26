from src.trainers.train_rocket import train_rocket

train_rocket(agent_type = 'SAC',
             number_of_episodes = 200,
             save_interval = 3,
             info = 'test',
             marl_load_info = None,
             load_network = True,
             debug_mode = False)