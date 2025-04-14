from configs.evolutionary_algorithms_config import subsonic_pso_params, supersonic_pso_params, flip_over_pso_params
from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation
from src.envs.pso.env_wrapped_ea import pso_wrapped_env
import datetime

flight_stage = 'subsonic' # 'subsonic' or 'supersonic' or 'flip_over'

if flight_stage == 'subsonic':
    pso_params = subsonic_pso_params
    model_name = 'subsonic_ascent'
elif flight_stage == 'supersonic':
    pso_params = supersonic_pso_params
    model_name = 'supersonic_ascent'
elif flight_stage == 'flip_over':
    model_name = 'flip_over'
    pso_params = flip_over_pso_params
else:
    raise ValueError(f"Flight stage {flight_stage} not supported")

run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
particle_swarm_optimisation = ParticleSubswarmOptimisation(pso_params = pso_params,
                                                           model = pso_wrapped_env(model_name = model_name,
                                                                                   sizing_needed_bool = False,
                                                                                   run_id = run_id,
                                                                                   flight_stage = flight_stage),
                                                           model_name = model_name,
                                                           run_id = run_id,
                                                           save_interval = 5)
particle_swarm_optimisation()

# To run tensorboard:
# cd data/pso_saves/ascent_agent
# tensorboard --logdir=runs

# To view on another device:
# ngrok http 6006