import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm                                                                                                                       


class ParticleSwarmOptimization:
    def __init__(self, pso_params, bounds, model, model_name):
        self.pop_size = pso_params['pop_size']
        self.generations = pso_params['generations']
        self.w_start = pso_params['w_start']
        self.w_end = pso_params['w_end']
        self.c1 = pso_params['c1']
        self.c2 = pso_params['c2']

        self.bounds = bounds
        self.model = model
        self.model_name = model_name
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
    
    def run(self):
        # Create tqdm progress bar with dynamic description
        pbar = tqdm(range(self.generations), desc='Running Particle Swarm Optimisation')
        
        for generation in pbar:
            for particle in self.swarm:
                fitness = self.evaluate_particle(particle)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()
            
            self.w = self.weight_linear_decrease(generation)
            for particle in self.swarm:
                self.update_velocity(particle, self.global_best_position)
                self.update_position(particle)

            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Swarm Optimisation - Best Fitness: {self.global_best_fitness:.2e}")
        
        return self.global_best_position, self.global_best_fitness
    
    def plot_results(self):
        self.model.plot_results(self.global_best_position)

    def plot_convergence(self, model_name):
        generations = range(len(self.global_best_fitness_array))

        file_path = f'results/{model_name}/particle_swarm_optimisation/convergence.png'

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(generations, self.global_best_fitness_array, linewidth=2)
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Best Fitness', fontsize=16)
        plt.title('Particle Swarm Optimisation Convergence', fontsize=18)
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

class ParticleSwarmOptimization_Subswarms(ParticleSwarmOptimization):
    def __init__(self, pso_params, bounds, model, model_name):
        super().__init__(pso_params, bounds, model, model_name)
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

    def run(self):
        # Create tqdm progress bar with dynamic description
        pbar = tqdm(range(self.generations), desc='Particle Swarm Optimisation with Subswarms')
        
        for generation in pbar:
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
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            if generation % 5 == 0:
                self.model.plot_results(self.global_best_position,
                                self.model_name,
                                'particle_subswarm_optimisation')

                self.plot_convergence(self.model_name)

            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Subswarm Optimisation - Best Fitness: {self.global_best_fitness:.6e}")

        return self.global_best_position, self.global_best_fitness
    
    def plot_convergence(self, model_name):
        generations = range(len(self.global_best_fitness_array))

        file_path = f'results/{model_name}/particle_subswarm_optimisation/convergence.png'

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(generations, self.global_best_fitness_array, linewidth=2)
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Best Fitness', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()