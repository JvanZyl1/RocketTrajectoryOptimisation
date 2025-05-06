from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation

flight_phase = 're_entry_burn' # 'subsonic' or 'supersonic' or 'flip_over_boostbackburn' or 'ballistic_arc_descent'
particle_swarm_optimisation = ParticleSubswarmOptimisation(flight_phase= flight_phase,
                                                           save_interval = 5,
                                                           enable_wind = False)
particle_swarm_optimisation()

# To run tensorboard:
# cd data/pso_saves/{flight_phase}
# tensorboard --logdir=runs

# To view on another device:
# ngrok http 6006