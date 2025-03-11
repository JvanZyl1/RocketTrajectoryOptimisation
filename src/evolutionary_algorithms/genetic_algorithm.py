import numpy as np
import matplotlib.pyplot as plt
import random

class GeneticAlgorithm:
    def __init__(self,
                 GA_params,
                 bounds,
                 model):
        self.population_size = GA_params['population_size']
        self.generations = GA_params['generations']
        self.crossover_rate = GA_params['crossover_rate']
        self.mutation_rate = GA_params['mutation_rate']
        self.fitness_threshold = GA_params['fitness_threshold']
        self.elite_size = GA_params.get('elite_size', 2)  # Default to 2 elite individuals if not specified

        self.bounds = bounds
        self.model = model

        self.population = self.initialize_population()
        self.fitness_scores = []

        self.best_fitness = None
        self.best_individual = None
        self.best_fitness_array = []
        self.best_individual_array = []

    def reset(self):
        self.population = self.initialize_population()
        self.fitness_scores = []
        self.best_fitness = None
        self.best_individual = None
        self.best_fitness_array = []
        self.best_individual_array = []

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            for b in self.bounds:
                value = random.uniform(b[0], b[1])
                individual.append(value)
            population.append(individual)
        return population

    def evaluate_population(self):
        self.fitness_scores = []  # Reset the fitness scores
        for individual in self.population:
            fitness = self.model.objective_function(individual)
            self.model.reset()
            self.fitness_scores.append(fitness)

    def select_parents(self):
        total_fitness = sum(1/f for f in self.fitness_scores if f != 0)  # Avoid division by zero

        # Calculate selection probabilities inversely proportional to fitness
        if total_fitness == 0:
            selection_probs = [1/len(self.fitness_scores)] * len(self.fitness_scores)
        else:
            selection_probs = [(1/f)/total_fitness if f != 0 else 0 for f in self.fitness_scores]

        selected_parents = random.choices(
            self.population,
            weights=selection_probs,
            k=self.population_size - self.elite_size  # Adjusted for elitism
        )

        return selected_parents
    
    def crossover(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[min(i+1, len(parents)-1)]
            if random.random() < self.crossover_rate:
                crossover_point = random.randint(1, len(parent1)-1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1, parent2
            offspring.extend([child1, child2])
        return offspring
    
    def mutate(self, offspring):
        for individual in offspring:
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
        return offspring

    def run_genetic_algorithm(self, print_bool=False):
        self.initialize_population()

        best_fitness_array = []
        best_individual_array = []
        
        for generation in range(self.generations):
            self.evaluate_population()
            
            # Sort population based on fitness (ascending order)
            sorted_population = [x for _, x in sorted(zip(self.fitness_scores, self.population))]
            sorted_fitness_scores = sorted(self.fitness_scores)

            # Select the elite individuals
            elites = sorted_population[:self.elite_size]

            # Perform selection, crossover, and mutation on the rest
            parents = self.select_parents()
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            
            # Combine elites with offspring to form the new population
            self.population = elites + offspring
            
            best_fitness = min(sorted_fitness_scores)
            best_individual = sorted_population[sorted_fitness_scores.index(best_fitness)]
            best_fitness_array.append(best_fitness)
            best_individual_array.append(best_individual)
            
            if print_bool:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

            # Stop if the error is below a certain threshold
            if best_fitness < self.fitness_threshold:
                break

        self.best_individual = best_individual
        self.best_fitness = best_fitness
        self.best_fitness_array = best_fitness_array
        self.best_individual_array = best_individual_array

        self.model.plot_results(self.best_individual)

        return best_individual, best_fitness
        
    def plot_convergence(self):
        generations = range(self.generations)

        plt.figure(figsize=(10, 10))
        # (2,1) subplot
        plt.subplot(2, 1, 1)
        plt.plot(generations, self.best_fitness_array)
        plt.xlabel('Generation [-]')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Convergence')
        # Now one with a log scale for y
        plt.subplot(2, 1, 2)
        plt.plot(generations, self.best_fitness_array)
        plt.yscale('log')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Convergence (Log Scale)')
        #plt.show()


class IslandGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, GA_params, bounds, model):
        super().__init__(GA_params, bounds, model)
        self.num_islands = GA_params['num_islands']
        self.num_migrants = GA_params['num_migrants']
        self.migration_interval = GA_params['migration_interval']
        self.generations = GA_params['generations']

        self.GA_params = GA_params

        self.islands = []

        self.best_fitness_array = []
        self.best_individual_array = []

        # check if the population size is enough for migration
        if not self.population_size_enough_for_migration():
            raise ValueError("Population size should be at least twice the number of migrants")

    def population_size_enough_for_migration(self):
        # Total population size should be at least twice the number of migrants
        return self.population_size >= 2 * self.num_migrants

    def migrate(self):
        for i in range(len(self.islands)):
            # Select the next island in the list, wrapping around to the first island
            next_island = self.islands[(i + 1) % len(self.islands)]
            
            # Select individuals randomly from the current island to migrate
            migrants = random.sample(self.islands[i].population, self.num_migrants)
            
            # Remove selected migrants from the current island
            self.islands[i].population = [ind for ind in self.islands[i].population if ind not in migrants]
            
            # Add migrants to the next island's population
            next_island.population.extend(migrants)

        return self.islands
    
    def run_island_genetic_algorithm(self,print_bool=False):
        # Initialise islands i,e. an instance of the GeneticAlgorithm class for each island
        for island in range(self.num_islands):
            self.islands.append(GeneticAlgorithm(self.GA_params, self.bounds, self.model))
        for generation in range(self.generations):
            for island in self.islands:
                island.run_genetic_algorithm()

            if generation % self.migration_interval == 0:
                self.islands = self.migrate()

            # Find the best fitness value among all islands
            best_fitness_values = [island.best_fitness for island in self.islands]
            best_fitness = min(best_fitness_values)

            # Find the index of the island with the best fitness value
            best_fitness_index = best_fitness_values.index(best_fitness)

            # Get the best individual from the island with the best fitness
            best_individual = self.islands[best_fitness_index].best_individual

            if print_bool:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

            # Append the best fitness and individual from each island to the arrays
            self.best_fitness_array.append(best_fitness)
            self.best_individual_array.append(best_individual)

            # Stop if the error is below a certain threshold
            if best_fitness < self.fitness_threshold:
                break

        for island in self.islands:
                self.best_fitness = island.best_fitness
                self.best_individual = island.best_individual

        return self.best_individual, self.best_fitness