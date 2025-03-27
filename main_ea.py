from configs.evolutionary_algorithms_config import pso_params
from src.evolutionary_algorithms.particle_swarm_optimisation import ParticleSubswarmOptimisation
from src.envs.pso.env_wrapped_ea import pso_wrapped_env

particle_swarm_optimisation = ParticleSubswarmOptimisation(pso_params = pso_params,
                                                           model = pso_wrapped_env(),
                                                           model_name = 'ascent_agent',
                                                           save_interval = 5)
particle_swarm_optimisation()

# To run tensorboard:
# cd data/pso_saves/ascent_agent
# tensorboard --logdir=runs

# To view on another device:
# ngrok http 6006