from src.trainers.train_rocket import train_rocket

train_rocket(agent_type = 'SAC',
             number_of_episodes = 2000,
             save_interval = 5,
             info = 'test',
             load_network = True,
             critic_warm_up_steps = 600,
             flight_stage = 'subsonic')