from src.evolutionary_algorithms.evolutionary_algorithms import EvolutionaryAlgorithms
from configs.evolutionary_algorithms_config import genetic_algorithm_params, pso_params
from src.evolutionary_algorithms.env_EA import env_EA_endo_ascent

def endo_ascent_EA(print_bool = False):
    
    model_name = 'endo_ascent_EA_fitting'
    model = env_EA_endo_ascent()
    evolutionary_algorithms = EvolutionaryAlgorithms(genetic_algorithm_params,
                                                     pso_params,
                                                     model,
                                                     model_name,
                                                     print_bool)
    evolutionary_algorithms.run_PSO()

    return evolutionary_algorithms.results