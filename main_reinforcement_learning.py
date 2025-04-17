from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC

trainer = RocketTrainer_SAC(flight_phase = 'subsonic',
                             load_from = 'None',
                             save_interval = 5,
                             pre_train_critic_bool = False)
trainer()