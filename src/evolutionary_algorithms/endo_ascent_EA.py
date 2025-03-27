from src.evolutionary_algorithms.evolutionary_algorithms import EvolutionaryAlgorithms
from configs.evolutionary_algorithms_config import genetic_algorithm_params, pso_params
from src.evolutionary_algorithms.env_EA import env_EA_endo_ascent

def endo_ascent_EA(algorithm_name, print_bool = True):    
    model_name = 'endo_ascent_EA_fitting'
    model = env_EA_endo_ascent()
    evolutionary_algorithms = EvolutionaryAlgorithms(genetic_algorithm_params,
                                                     pso_params,
                                                     model,
                                                     model_name,
                                                     print_bool)
    if algorithm_name == 'genetic_algorithm':
        evolutionary_algorithms.run_genetic_algorithm()
    elif algorithm_name == 'island_genetic_algorithm':
        evolutionary_algorithms.run_island_genetic_algorithm()
    elif algorithm_name == 'particle_swarm_optimisation':
        evolutionary_algorithms.run_particle_swarm_optimisation()
    elif algorithm_name == 'particle_subswarm_optimisation':
        evolutionary_algorithms.run_particle_subswarm_optimisation()
    else:
        evolutionary_algorithms.run_evolutionary_algorithms()


    return evolutionary_algorithms.results