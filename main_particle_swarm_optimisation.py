from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation

flight_phase = 'flip_over_boostbackburn' # 'subsonic' or 'supersonic' or 'flip_over_boostbackburn'
particle_swarm_optimisation = ParticleSubswarmOptimisation(save_interval = 5)
particle_swarm_optimisation()

# To run tensorboard:
# cd data/pso_saves/ascent_agent
# tensorboard --logdir=runs

# To view on another device:
# ngrok http 6006