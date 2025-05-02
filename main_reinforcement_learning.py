from src.trainers.trainer_rocket_SAC import RocketTrainer_ReinforcementLearning

trainer = RocketTrainer_ReinforcementLearning(flight_phase = 'flip_over_boostbackburn',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 10,
                             pre_train_critic_bool = False,
                             buffer_type = 'uniform',
                             rl_type = 'td3',
                             enable_wind = False)
trainer()