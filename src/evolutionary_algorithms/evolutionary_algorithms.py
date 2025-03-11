import matplotlib.pyplot as plt
import pandas as pd

from src.evolutionary_algorithms.genetic_algorithm import GeneticAlgorithm, IslandGeneticAlgorithm
from src.evolutionary_algorithms.particle_swarm_optimisation import ParticleSwarmOptimization, ParticleSwarmOptimization_Subswarms

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
        
        self.genetic_algorithm = GeneticAlgorithm(self.genetic_algorithm_params, self.model.bounds, self.model)
        self.island_genetic_algorithm = IslandGeneticAlgorithm(self.genetic_algorithm_params, self.model.bounds, self.model)
        self.particle_swarm_optimisation = ParticleSwarmOptimization(self.pso_params, self.model.bounds, self.model)
        self.particle_subswarm_optimisation = ParticleSwarmOptimization_Subswarms(self.pso_params, self.model.bounds, self.model)

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
            }
        }

        self.algorithm_key = None # genetic_algorithm, island_genetic_algorithm, particle_swarm_optimisation, particle_subswarm_optimisation as string
        self.formatted_results = None

    def plot_results_evolutionary_algorithm(self,
                                            optimisation_parameters,
                                            evolutionary_algorithm):
        self.model.individual_update_model(optimisation_parameters)
        file_path = f'results/{self.model_name}/{self.algorithm_key}/SimulationResults.png'
        self.model.plot_results(optimisation_parameters)
        plt.savefig(file_path)
        plt.close()

        file_path = f'results/{self.model_name}/{self.algorithm_key}/Convergence.png'
        evolutionary_algorithm.plot_convergence()
        plt.savefig(file_path)
        plt.close()

    def run_GA(self):
        print("Running Genetic Algorithm")
        best_solution_GA, best_value_GA = self.genetic_algorithm.run_genetic_algorithm(print_bool=self.print_bool)
        
        self.results['genetic_algorithm']['best_solution'] = best_solution_GA
        self.results['genetic_algorithm']['best_value'] = best_value_GA

        self.algorithm_key = 'genetic_algorithm'
        self.plot_results_evolutionary_algorithm(best_solution_GA, self.genetic_algorithm)
        self.update_results_file()

    def run_IGA(self):
        print("Running Island Genetic Algorithm")
        best_solution_IGA, best_value_IGA = self.island_genetic_algorithm.run_island_genetic_algorithm(print_bool=True)
        
        self.results['island_genetic_algorithm']['best_solution'] = best_solution_IGA
        self.results['island_genetic_algorithm']['best_value'] = best_value_IGA

        self.algorithm_key = 'island_genetic_algorithm'
        self.plot_results_evolutionary_algorithm(best_solution_IGA, self.island_genetic_algorithm)
        self.update_results_file()

    def run_PSO(self):
        print("Runing Particle Swarm Optimization")
        best_solution_PSO, best_value_PSO = self.particle_swarm_optimisation.run(print_bool=True)
        self.results['particle_swarm_optimisation']['best_solution'] = best_solution_PSO
        self.results['particle_swarm_optimisation']['best_value'] = best_value_PSO

        self.algorithm_key = 'particle_swarm_optimisation'
        self.plot_results_evolutionary_algorithm(best_solution_PSO, self.particle_swarm_optimisation)
        self.update_results_file()

    def run_PSO_subswarms(self):
        print("Running Particle Swarm Optimization with Subswarms")
        best_solution_PSO_subswarms, best_value_PSO_subswarms = self.particle_subswarm_optimisation.run(print_bool=True)

        self.results['particle_subswarm_optimisation']['best_solution'] = best_solution_PSO_subswarms
        self.results['particle_subswarm_optimisation']['best_value'] = best_value_PSO_subswarms

        self.algorithm_key = 'particle_subswarm_optimisation'
        self.plot_results_evolutionary_algorithm(best_solution_PSO_subswarms, self.particle_subswarm_optimisation)
        self.update_results_file()

    def run_evolutionary_algorithms(self):
        self.run_GA()
        self.genetic_algorithm.reset()
        self.run_IGA()
        self.island_genetic_algorithm.reset()
        self.run_PSO()
        self.particle_swarm_optimisation.reset()
        self.run_PSO_subswarms()
        self.particle_subswarm_optimisation.reset()
        self.end_print()

    def update_results_file(self):
        file_path = f'results/{self.model_name}/data.txt'
        # Read the file to get the other results
        with open(file_path, 'r') as file:
            lines = file.readlines()

        opt_params_list = list(self.mock_dictionary_of_opt_params.values())
        column_titles = list(self.mock_dictionary_of_opt_params.keys()) + ['Best Fitness']

        # If file is empty set up column and row titles, fill the rest with 10e10
        if len(lines) == 0:
            row_titles = ['GA_results    ', 'Island_GA_results    ', 'PSO_results    ', 'PSO_with_Subswarms    ']
            empty_row = opt_params_list + [10e10]

            header = ' '.join(column_titles)

            with open(file_path, 'w') as file:
                file.write(header + '\n')

            mock_params = [10e10] * len(opt_params_list)

            results = [
                ['GA_results', mock_params, 10e10],
                ['Island_GA_results', mock_params, 10e10],
                ['PSO_results', mock_params, 10e10],
                ['PSO_with_Subswarms', mock_params,10e10]
            ]

            df_list = []
            for result in results:
                method_name, solution, score = result
                row_data = list(solution) + [score]
                df = pd.DataFrame([row_data], columns=column_titles, index=[method_name])
                df_list.append(df)

            # Concatenate all DataFrames
            final_df = pd.concat(df_list)

            # Convert DataFrame to string with tabulate or simply use to_string() for simplicity
            formatted_results = final_df.to_string()

            with open(file_path, 'w') as file:
                file.write(formatted_results)

             # Read the file to get the other results
            with open(file_path, 'r') as file:
                lines = file.readlines()


        # Extract the results from the lines and split into "name", "solution", "value"
        results = []
        line_counter = 0
        for line in lines:
            if line_counter == 0:
                line_counter += 1
            else:
                line = line.strip()
                name, _, solution_value = line.split(' ', 2)
                # Turn solution and value into list of floats
                solution_value = solution_value.split()

                def parse_value(val):
                    try:
                        return float(val)
                    except ValueError:
                        return 10e10

                solution = [parse_value(val) for val in solution_value[:-1]]
                value = parse_value(solution_value[-1])
                results.append([name, solution, value])

        if self.algorithm_key == 'genetic_algorithm':
            best_solution = self.results['genetic_algorithm']['best_solution']
            best_value = self.results['genetic_algorithm']['best_value']
            # Update the results list with the new results
            results[0] = ["GA_results", best_solution, best_value]

        elif self.algorithm_key == 'island_genetic_algorithm':
            best_solution = self.results['island_genetic_algorithm']['best_solution']
            best_value = self.results['island_genetic_algorithm']['best_value']
            results[1] = ["Island_GA_results", best_solution, best_value]
        
        elif self.algorithm_key == 'particle_swarm_optimisation':
            best_solution = self.results['particle_swarm_optimisation']['best_solution']
            best_value = self.results['particle_swarm_optimisation']['best_value']
            results[2] = ["PSO_results", best_solution, best_value]

        elif self.algorithm_key == 'particle_subswarm_optimisation':
            best_solution = self.results['particle_subswarm_optimisation']['best_solution']
            best_value = self.results['particle_subswarm_optimisation']['best_value']
            results[3] = ["PSO_with_Subswarms", best_solution, best_value]

        df_list = []
        for result in results:
            method_name, solution, score = result
            row_data = list(solution) + [score]
            df = pd.DataFrame([row_data], columns=column_titles, index=[method_name])
            df_list.append(df)

        # Concatenate all DataFrames
        final_df = pd.concat(df_list)

        # Convert DataFrame to string with tabulate or simply use to_string() for simplicity
        formatted_results = final_df.to_string()

        with open(file_path, 'w') as file:
            file.write(formatted_results)

        self.formatted_results = formatted_results

    def end_print(self):
        print("Results for the best parameters for each algorithm:")
        print(self.formatted_results)