from src.trainers.trainer_rocket_SAC import RocketTrainer_ReinforcementLearning

trainer = RocketTrainer_ReinforcementLearning(flight_phase = 'landing_burn',
                             load_from = None,
                             load_buffer_bool= False,
                             save_interval = 5,
                             pre_train_critic_bool = False,
                             buffer_type = 'uniform',
                             rl_type = 'sac',
                             enable_wind = False)
trainer()