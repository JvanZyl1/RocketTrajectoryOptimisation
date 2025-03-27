from src.evolutionary_algorithms.endo_ascent_EA import endo_ascent_EA
import os

algorithm_name = 'particle_subswarm_optimisation' # genetic_algorithm, island_genetic_algorithm,
                                     # particle_swarm_optimisation, particle_subswarm_optimisation,
                                     # all

# Create logging directory
log_dir = f"data/pso_saves/endo_ascent_EA_fitting/{algorithm_name}"
os.makedirs(log_dir, exist_ok=True)

_ = endo_ascent_EA(algorithm_name)

# To run tensorboard: tensorboard --logdir={LOCATION_OF_LOG_DIR}