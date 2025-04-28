from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC

trainer = RocketTrainer_SAC(flight_phase = 'subsonic',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 10,
                             pre_train_critic_bool = False,
                             enable_wind = True)
trainer()   
'''
from src.trainers.trainer_rocket_SAC_vectorized import RocketTrainer_SAC_Vectorized
trainer = RocketTrainer_SAC_Vectorized(flight_phase = 'subsonic',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 5,
                             pre_train_critic_bool = False,
                             num_parallel_envs = 1,
                             enable_wind = True)
trainer()
'''