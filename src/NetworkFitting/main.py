from src.NetworkFitting.particle_swarm_optimisation import ParticleSwarmOptimization_Subswarms
from src.NetworkFitting.env import env_ea
from configs.config import pso_params

def run_network_fitting():
    # First fit for zero-alpha
    model_zero_alpha = env_ea(zero_alpha_bool = True)
    particle_subswarm_optimiser_zero_alpha = ParticleSwarmOptimization_Subswarms(pso_params = pso_params,
                                                                                 bounds = model_zero_alpha.bounds,
                                                                                 model = model_zero_alpha,
                                                                                 model_name = 'zero_alpha')

    best_individual_forces_network, _ = particle_subswarm_optimiser_zero_alpha.run()

    # Then fit for non-zero alpha
    model_non_zero_alpha = env_ea(zero_alpha_bool = False,
                                  force_network_individual = best_individual_forces_network)
    particle_subswarm_optimiser_non_zero_alpha = ParticleSwarmOptimization_Subswarms(pso_params = pso_params,
                                                                                     bounds = model_non_zero_alpha.bounds,
                                                                                     model = model_non_zero_alpha,
                                                                                     model_name = 'non_zero_alpha')
    
    best_individual_moment_and_force_network, _ = particle_subswarm_optimiser_non_zero_alpha.run()