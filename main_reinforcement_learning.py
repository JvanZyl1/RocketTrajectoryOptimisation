from src.trainers.trainer_rocket_SAC import RocketTrainer_SAC
from src.trainers.trainer_rocket_SAC_vectorized import test_vectorized_performance

test_vectorized_performance()

'''
trainer = RocketTrainer_SAC(flight_phase = 'subsonic',
                             load_from = 'supervisory',
                             load_buffer_bool= False,
                             save_interval = 10,
                             pre_train_critic_bool = False)
trainer()   
'''