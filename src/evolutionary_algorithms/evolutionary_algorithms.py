import matplotlib.pyplot as plt
import pandas as pd

from src.evolutionary_algorithms.genetic_algorithm import GeneticAlgorithm, IslandGeneticAlgorithm
from src.evolutionary_algorithms.particle_swarm_optimisation import ParticleSwarmOptimization, ParticleSwarmOptimization_Subswarms
from src.evolutionary_algorithms.local_search_optimisers import ParticleSwarmOptimization_with_local_search

class EvolutionaryAlgorithms():
    def __init__(self,
                 genetic_algorithm_params,
                 pso_params,
                 model,
                 model_name,
                 print_bool = False):
        self.print_bool = print_bool
        self.model = model
        self.model_name = model_name

        self.mock_dictionary_of_opt_params = self.model.mock_dictionary_of_opt_params

        self.genetic_algorithm_params = genetic_algorithm_params
        self.pso_params = pso_params
        
        self.genetic_algorithm = GeneticAlgorithm(self.genetic_algorithm_params, self.model.bounds, self.model, self.model_name)
        self.island_genetic_algorithm = IslandGeneticAlgorithm(self.genetic_algorithm_params, self.model.bounds, self.model, self.model_name)
        self.particle_swarm_optimisation = ParticleSwarmOptimization(self.pso_params, self.model.bounds, self.model, self.model_name)
        self.particle_subswarm_optimisation = ParticleSwarmOptimization_Subswarms(self.pso_params, self.model.bounds, self.model, self.model_name)
        self.particle_swarm_optimisation_with_local_search = ParticleSwarmOptimization_with_local_search(self.pso_params, self.model.bounds, self.model, self.model_name)
        self.results = {
            'genetic_algorithm' : {
                'best_solution' : None,
                'best_value' : None
            },
            'island_genetic_algorithm' : {
                'best_solution' : None,
                'best_value' : None
            },
            'particle_swarm_optimisation' : {
                'best_solution' : None,
                'best_value' : None
            },
            'particle_subswarm_optimisation' : {
                'best_solution' : None,
                'best_value' : None
            },
            'particle_swarm_optimisation_with_local_search' : {
                'best_solution' : None,
                'best_value' : None
            }
        }

        self.algorithm_key = None # genetic_algorithm, island_genetic_algorithm, particle_swarm_optimisation, particle_subswarm_optimisation as string
        self.formatted_results = None

    def plot_results_evolutionary_algorithm(self,
                                            optimisation_parameters,
                                            evolutionary_algorithm):
        self.model.plot_results(optimisation_parameters,
                                self.model_name,
                                self.algorithm_key)

        evolutionary_algorithm.plot_convergence(self.model_name)

    def run_genetic_algorithm(self):
        best_solution_genetic_algorithm, best_value_genetic_algorithm = self.genetic_algorithm.run_genetic_algorithm()
        
        self.results['genetic_algorithm']['best_solution'] = best_solution_genetic_algorithm
        self.results['genetic_algorithm']['best_value'] = best_value_genetic_algorithm

        self.algorithm_key = 'genetic_algorithm'
        self.plot_results_evolutionary_algorithm(best_solution_genetic_algorithm, self.genetic_algorithm)
        self.update_results_file()

    def run_island_genetic_algorithm(self):
        best_solution_island_genetic_algorithm, best_value_island_genetic_algorithm = self.island_genetic_algorithm.run_island_genetic_algorithm()
        
        self.results['island_genetic_algorithm']['best_solution'] = best_solution_island_genetic_algorithm
        self.results['island_genetic_algorithm']['best_value'] = best_value_island_genetic_algorithm

        self.algorithm_key = 'island_genetic_algorithm'
        self.plot_results_evolutionary_algorithm(best_solution_island_genetic_algorithm, self.island_genetic_algorithm)
        self.update_results_file()

    def run_particle_swarm_optimisation(self):
        best_solution_particle_swarm_optimisation, best_value_particle_swarm_optimisation = self.particle_swarm_optimisation.run()
        self.results['particle_swarm_optimisation']['best_solution'] = best_solution_particle_swarm_optimisation
        self.results['particle_swarm_optimisation']['best_value'] = best_value_particle_swarm_optimisation

        self.algorithm_key = 'particle_swarm_optimisation'
        self.plot_results_evolutionary_algorithm(best_solution_particle_swarm_optimisation, self.particle_swarm_optimisation)
        self.update_results_file()

    def run_particle_subswarm_optimisation(self):
        best_solution_particle_subswarm_optimisation, best_value_particle_subswarm_optimisation = self.particle_subswarm_optimisation.run()

        self.results['particle_subswarm_optimisation']['best_solution'] = best_solution_particle_subswarm_optimisation
        self.results['particle_subswarm_optimisation']['best_value'] = best_value_particle_subswarm_optimisation

        self.algorithm_key = 'particle_subswarm_optimisation'
        self.plot_results_evolutionary_algorithm(best_solution_particle_subswarm_optimisation, self.particle_subswarm_optimisation)
        self.update_results_file()

    def run_particle_swarm_optimisation_with_local_search(self):
        best_solution_particle_swarm_optimisation_with_local_search, best_value_particle_swarm_optimisation_with_local_search = self.particle_swarm_optimisation_with_local_search.run()

        self.results['particle_swarm_optimisation_with_local_search']['best_solution'] = best_solution_particle_swarm_optimisation_with_local_search
        self.results['particle_swarm_optimisation_with_local_search']['best_value'] = best_value_particle_swarm_optimisation_with_local_search

        self.algorithm_key = 'particle_swarm_optimisation_with_local_search'
        self.plot_results_evolutionary_algorithm(best_solution_particle_swarm_optimisation_with_local_search, self.particle_swarm_optimisation_with_local_search)
        self.update_results_file()

    def run_evolutionary_algorithms(self):
        self.run_genetic_algorithm()
        self.genetic_algorithm.reset()
        self.run_island_genetic_algorithm()
        self.island_genetic_algorithm.reset()
        self.run_particle_swarm_optimisation()
        self.particle_swarm_optimisation.reset()
        self.run_particle_subswarm_optimisation()
        self.particle_subswarm_optimisation.reset()
        self.run_particle_swarm_optimisation_with_local_search()
        self.particle_swarm_optimisation_with_local_search.reset()
        #self.end_print()

    def update_results_file(self):
        # Change file extension from txt to csv
        file_path = f'results/{self.model_name}/evolutionary_results.csv'
        
        column_titles = list(self.mock_dictionary_of_opt_params.keys()) + ['Best Fitness']
        
        # Check if file exists and is not empty
        try:
            # Try to read existing CSV file
            existing_df = pd.read_csv(file_path, index_col=0)
            file_exists = True
        except (FileNotFoundError, pd.errors.EmptyDataError):
            file_exists = False
        
        # If file doesn't exist or is empty, create a new DataFrame with mock data
        if not file_exists:
            # Make sure mock_params has the same length as opt_params_list
            mock_params = [10e10] * (len(column_titles) - 1)  # Subtract 1 for 'Best Fitness'
            
            # Create initial DataFrame with algorithm names as rows
            data = []
            algorithms = ['Genetic Algorithm', 'Island Genetic Algorithm', 
                         'Particle Swarm Optimisation', 'Particle Subswarm Optimisation',
                         'Particle Swarm Optimisation with Local Search']
            
            for algorithm in algorithms:
                row = [algorithm] + mock_params + [10e10]
                data.append(row)
            
            # Create DataFrame with an Algorithm column
            df_columns = ['Algorithm'] + column_titles
            existing_df = pd.DataFrame(data, columns=df_columns)
            existing_df.set_index('Algorithm', inplace=True)
            existing_df.to_csv(file_path)
        
        # Update the DataFrame with new results based on algorithm_key
        if self.algorithm_key == 'genetic_algorithm':
            best_solution = self.results['genetic_algorithm']['best_solution']
            best_value = self.results['genetic_algorithm']['best_value']
            row_data = dict(zip(column_titles[:-1], best_solution))
            row_data[column_titles[-1]] = best_value
            existing_df.loc['Genetic Algorithm'] = row_data
            
        elif self.algorithm_key == 'island_genetic_algorithm':
            best_solution = self.results['island_genetic_algorithm']['best_solution']
            best_value = self.results['island_genetic_algorithm']['best_value']
            row_data = dict(zip(column_titles[:-1], best_solution))
            row_data[column_titles[-1]] = best_value
            existing_df.loc['Island Genetic Algorithm'] = row_data
        
        elif self.algorithm_key == 'particle_swarm_optimisation':
            best_solution = self.results['particle_swarm_optimisation']['best_solution']
            best_value = self.results['particle_swarm_optimisation']['best_value']
            row_data = dict(zip(column_titles[:-1], best_solution))
            row_data[column_titles[-1]] = best_value
            existing_df.loc['Particle Swarm Optimisation'] = row_data
            
        elif self.algorithm_key == 'particle_subswarm_optimisation':
            best_solution = self.results['particle_subswarm_optimisation']['best_solution']
            best_value = self.results['particle_subswarm_optimisation']['best_value']
            row_data = dict(zip(column_titles[:-1], best_solution))
            row_data[column_titles[-1]] = best_value
            existing_df.loc['Particle Subswarm Optimisation'] = row_data
        
        elif self.algorithm_key == 'particle_swarm_optimisation_with_local_search':
            best_solution = self.results['particle_swarm_optimisation_with_local_search']['best_solution']
            best_value = self.results['particle_swarm_optimisation_with_local_search']['best_value']
            row_data = dict(zip(column_titles[:-1], best_solution))
            row_data[column_titles[-1]] = best_value
            existing_df.loc['Particle Swarm Optimisation with Local Search'] = row_data
            
        # Save the updated DataFrame to CSV
        existing_df.to_csv(file_path)
        
        # Store formatted results for display
        self.formatted_results = existing_df.to_string()

    def end_print(self):
        print("Results for the best parameters for each algorithm:")
        print(self.formatted_results)