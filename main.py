#from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent

#env = rocket_model_endo_ascent(sizing_needed_bool = False)
#env.run_test_physics()


from src.trainers.train_rocket import train_rocket

train_rocket(agent_type = 'SAC',
             number_of_episodes = 100,
             save_interval = 4,
             info = 'test',
             marl_load_info = None)