from src.trainers.train_rocket import train_rocket

train_rocket(agent_type = 'SAC',
             number_of_episodes = 200,
             save_interval = 5,
             info = 'test',
             load_network = True,
             critic_warm_up_steps = 25000)