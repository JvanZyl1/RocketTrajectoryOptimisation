from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC

trainer = RocketTrainer_SAC(flight_phase = 'subsonic',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 5,
                             pre_train_critic_bool = False,
                             enable_wind = True)
trainer()