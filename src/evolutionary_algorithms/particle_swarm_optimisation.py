import random
import numpy as np
import matplotlib.pyplot as plt


class ParticleSwarmOptimization:
    def __init__(self, pso_params, bounds, model):
        self.pop_size = pso_params['pop_size']
        self.generations = pso_params['generations']
        self.w_start = pso_params['w_start']
        self.w_end = pso_params['w_end']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']

        self.bounds = bounds
        self.model = model

        self.best_fitness_array = []
        self.best_individual_array = []

        self.initialize_swarm()   
        self.w = self.w_start     

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []

    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.initialize_swarm()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.pop_size):
            position_array = []
            for bound in self.bounds:
                position = random.uniform(bound[0], bound[1])
                position_array.append(position)
            particle = {
                'position': np.array(position_array),
                'velocity': np.zeros(len(self.bounds)),
                'best_position': None,
                'best_fitness': float('inf')
            }
            swarm.append(particle)
        self.swarm = swarm

    def evaluate_particle(self, particle):
        individual = particle['position']
        fitness = self.model.objective_function(individual)
        self.model.reset()
        if fitness < particle['best_fitness']:
            particle['best_fitness'] = fitness
            particle['best_position'] = particle['position'].copy()
        return fitness
    

    def update_velocity(self, particle, global_best_position):
        inertia = self.w * particle['velocity']
        cognitive = self.c1 * np.random.rand() * (particle['best_position'] - particle['position'])
        social = self.c2 * np.random.rand() * (global_best_position - particle['position'])
        particle['velocity'] = inertia + cognitive + social

    def update_position(self, particle):
        particle['position'] += particle['velocity']
        for i in range(len(self.bounds)):
            if particle['position'][i] < self.bounds[i][0]:
                particle['position'][i] = self.bounds[i][0]
            elif particle['position'][i] > self.bounds[i][1]:
                particle['position'][i] = self.bounds[i][1]

    def weight_linear_decrease(self, generation):
        return self.w_start - (self.w_start - self.w_end) * generation / self.generations
    
    def run(self, print_bool=True):
        for generation in range(self.generations):
            for particle in self.swarm:
                fitness = self.evaluate_particle(particle)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()
            
            self.w = self.weight_linear_decrease(generation)
            for particle in self.swarm:
                self.update_velocity(particle, self.global_best_position)
                self.update_position(particle)

            if print_bool:
                print(f"Generation {generation}: Best Fitness = {self.global_best_fitness}")
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)
        
        return self.global_best_position, self.global_best_fitness
    
    def plot_results(self):
        self.model.plot_results(self.global_best_position)

    def plot_convergence(self):
        generations = range(self.generations)

        # (2,1) subplot
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(generations, self.global_best_fitness_array)
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('PSO Convergence')

        # Now one with a log scale for y
        plt.subplot(2, 1, 2)
        plt.plot(generations, self.global_best_fitness_array)
        plt.yscale('log')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('PSO Convergence (Log Scale)')
        #plt.show()

class ParticleSwarmOptimization_Subswarms(ParticleSwarmOptimization):
    def __init__(self, pso_params, bounds, model):
        super().__init__(pso_params, bounds, model)
        self.num_sub_swarms = pso_params["num_sub_swarms"]
        self.initialize_swarms()  # Initializes multiple sub-swarms

    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.initialize_swarms()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []

    def initialize_swarms(self):
        self.swarms = []
        sub_swarm_size = self.pop_size // self.num_sub_swarms
        for _ in range(self.num_sub_swarms):
            swarm = []
            for _ in range(sub_swarm_size):
                position_array = [random.uniform(bound[0], bound[1]) for bound in self.bounds]
                particle = {
                    'position': np.array(position_array),
                    'velocity': np.zeros(len(self.bounds)),
                    'best_position': None,
                    'best_fitness': float('inf')
                }
                swarm.append(particle)
            self.swarms.append(swarm)

    def run(self, print_bool=True):
        for generation in range(self.generations):
            local_best_positions = []
            local_best_fitnesses = []

            # Evaluate each sub-swarm
            for swarm in self.swarms:
                local_best_fitness = float('inf')
                local_best_position = None

                for particle in swarm:
                    fitness = self.evaluate_particle(particle)
                    if fitness < particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()

                    if fitness < local_best_fitness:
                        local_best_fitness = fitness
                        local_best_position = particle['position'].copy()

                local_best_positions.append(local_best_position)
                local_best_fitnesses.append(local_best_fitness)

            # Update global best from local bests
            for i, fitness in enumerate(local_best_fitnesses):
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = local_best_positions[i].copy()

            self.w = self.weight_linear_decrease(generation)

            # Update particles in each sub-swarm
            for swarm in self.swarms:
                for particle in swarm:
                    self.update_velocity(particle, self.global_best_position)
                    self.update_position(particle)

            if print_bool:
                print(f"Generation {generation}: Best Fitness = {self.global_best_fitness}")
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

        return self.global_best_position, self.global_best_fitness