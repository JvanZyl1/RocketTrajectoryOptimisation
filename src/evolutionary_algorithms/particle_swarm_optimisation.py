import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

class ParticleSwarmOptimization:
    def __init__(self, pso_params, bounds, model, model_name, local_search_optimiser = None, local_search_plot_bool = False):
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
        self.average_particle_fitness_array = []

        # Create log directory if it doesn't exist
        if local_search_plot_bool is False:
            log_dir = f"data/pso_saves/{model_name}/particle_swarm_optimisation"
            os.makedirs(log_dir, exist_ok=True)
            self.local_search_plot_bool = False
        else:
            log_dir = f"data/pso_saves/{model_name}/particle_swarm_optimisation_with_local_search"
            self.local_search_plot_bool = True
            os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)

        # Initialise the non-heuristic optimiser
        self.local_search_number_of_particles = pso_params.get('local_search_number_of_particles', 10)
        self.local_search_frequency = pso_params.get('local_search_frequency', 4)
        self.local_search_optimiser = local_search_optimiser

    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.initialize_swarm()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []
        self.average_particle_fitness_array = []
        
        # Close the previous writer and create a new one
        self.writer.close()
        log_dir = f"runs/{self.model_name}/particle_swarm_optimisation"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
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
    
    def local_search_mutation(self):
        # 1) Select the top performing particles
        best_particles = sorted(self.swarm, key=lambda x: x['best_fitness'])[:self.local_search_number_of_particles]

        # 2) Run the non-heuristic optimiser on the best particles and update the swarm
        for particle in best_particles:
            particle['position'] = self.local_search_optimiser.run(particle['position'])
            particle['best_fitness'] = self.evaluate_particle(particle)
            particle['velocity'] = np.zeros(len(self.bounds))
            if particle['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = particle['best_fitness']
                self.global_best_position = particle['position'].copy()

        # 3) Replace the worst performing particles in the swarm with the new particles
        worst_particles = sorted(self.swarm, key=lambda x: x['best_fitness'])[-self.local_search_number_of_particles:]
        for i, particle in enumerate(worst_particles):
            self.swarm[i] = particle

        # 4) Return the updated swarm
        return self.swarm

    def run(self):
        # Create tqdm progress bar with dynamic description
        pbar = tqdm(range(self.generations), desc='Running Particle Swarm Optimisation')
        
        for generation in pbar:
            particle_fitnesses = []
            for i, particle in enumerate(self.swarm):
                fitness = self.evaluate_particle(particle)
                particle_fitnesses.append(fitness)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()

            average_particle_fitness = np.mean(particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)            
            
            if generation % self.local_search_frequency == 0 \
                and generation != 0 \
                and self.local_search_optimiser is not None:  # Run every 10 generations or at your preferred frequency
                self.local_search_mutation()

            self.w = self.weight_linear_decrease(generation)
            for i, particle in enumerate(self.swarm):
                self.update_velocity(particle, self.global_best_position)
                self.update_position(particle)

            
 
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            # Log metrics to TensorBoard
            self.writer.add_scalar('Fitness/Best', self.global_best_fitness, generation)
            self.writer.add_scalar('Fitness/Average', average_particle_fitness, generation)
            self.writer.add_scalar('Parameters/Inertia_Weight', self.w, generation)
            # Log particle positions as histograms
            for dim in range(len(self.bounds)):
                positions = [p['position'][dim] for p in self.swarm]
                self.writer.add_histogram(f'Particle_Positions/Dimension_{dim}', 
                                        np.array(positions), 
                                        generation)

            # Flush the writer periodically to ensure data is written
            if generation % 10 == 0 and generation != 0:
                self.writer.flush()

            if generation % 5 == 0 and generation != 0:
                self.model.plot_results(self.global_best_position,
                                self.model_name,
                                'particle_swarm_optimisation')
                for i, pos in enumerate(self.global_best_position):
                    self.writer.add_scalar(f'Best_Position/Dimension_{i}', pos, generation)

                self.plot_convergence(self.model_name)
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Swarm Optimisation - Best Fitness: {self.global_best_fitness:.4e}")
        
        # Make sure to flush at the end
        self.writer.flush()
        return self.global_best_position, self.global_best_fitness
    
    def plot_results(self):
        self.model.plot_results(self.global_best_position)

    def plot_convergence(self, model_name):
        generations = range(len(self.global_best_fitness_array))
    
        if self.local_search_plot_bool is False:
            file_path = f'results/{model_name}/particle_swarm_optimisation/convergence.png'
        else:
            file_path = f'results/{model_name}/particle_swarm_optimisation_with_local_search/convergence.png'

        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(generations, self.global_best_fitness_array, linewidth=2, label='Best Fitness')
        plt.plot(generations, self.average_particle_fitness_array, linewidth=2, label='Average Particle Fitness')
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle Swarm Optimisation Convergence', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        file_path = f'results/{model_name}/particle_swarm_optimisation/last_10_fitnesses.png'
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(self.global_best_fitness_array[-10:], linewidth=2, label='Best Fitness')
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle Swarm Optimisation Convergence', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def __del__(self):
        """Ensure the writer is closed when the object is deleted"""
        if hasattr(self, 'writer'):
            self.writer.close()



class ParticleSwarmOptimization_Subswarms(ParticleSwarmOptimization):
    def __init__(self, pso_params, bounds, model, model_name):
        super().__init__(pso_params, bounds, model, model_name)
        self.num_sub_swarms = pso_params["num_sub_swarms"]
        self.initialize_swarms()
        
        # Close the previous writer and create a new one with the correct path
        self.writer.close()
        log_dir = f"data/pso_saves/{model_name}/particle_subswarm_optimisation"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.initialize_swarms()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

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
        pbar = tqdm(range(self.generations), desc='Particle Swarm Optimisation with Subswarms')
        
        # Initialize arrays to track best fitness and average fitness for each subswarm
        self.subswarm_best_fitnesses = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_fitnesses = [[] for _ in range(self.num_sub_swarms)]
        
        for generation in pbar:
            local_best_positions = []
            local_best_fitnesses = []
            current_fitnesses = [[] for _ in self.swarms]

            # Evaluate each sub-swarm
            all_particle_fitnesses = []
            
            for swarm_idx, swarm in enumerate(self.swarms):
                local_best_fitness = float('inf')
                local_best_position = None

                for particle in swarm:
                    fitness = self.evaluate_particle(particle)
                    all_particle_fitnesses.append(fitness)
                    current_fitnesses[swarm_idx].append(fitness)
                    
                    if fitness < particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()

                    if fitness < local_best_fitness:
                        local_best_fitness = fitness
                        local_best_position = particle['position'].copy()

                # Apply backpropagation to the best particle in each subswarm
                if generation % self.backprop_freq == 0 and local_best_position is not None:
                    refined_position = self.backprop_refinement(local_best_position)
                    # Update the best particle in the swarm with the refined position
                    best_particle_idx = current_fitnesses[swarm_idx].index(local_best_fitness)
                    swarm[best_particle_idx]['position'] = refined_position
                    # Re-evaluate the refined particle
                    new_fitness = self.evaluate_particle(swarm[best_particle_idx])
                    if new_fitness < local_best_fitness:
                        local_best_fitness = new_fitness
                        local_best_position = refined_position

                local_best_positions.append(local_best_position)
                local_best_fitnesses.append(local_best_fitness)
                
                # Store best fitness for this subswarm
                self.subswarm_best_fitnesses[swarm_idx].append(local_best_fitness)

                # Calculate and store average fitness for this subswarm
                swarm_avg_fitness = np.mean(current_fitnesses[swarm_idx])
                self.subswarm_avg_fitnesses[swarm_idx].append(swarm_avg_fitness)

            average_particle_fitness = np.mean(all_particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)

            # Update global best from local bests
            for i, fitness in enumerate(local_best_fitnesses):
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = local_best_positions[i].copy()

            # Apply backpropagation to the global best position
            if generation % self.backprop_freq == 0 and self.global_best_position is not None:
                refined_global_position = self.backprop_refinement(self.global_best_position)
                # Re-evaluate the refined global position
                temp_particle = {'position': refined_global_position, 'velocity': np.zeros(len(self.bounds)), 
                                'best_position': None, 'best_fitness': float('inf')}
                new_global_fitness = self.evaluate_particle(temp_particle)
                if new_global_fitness < self.global_best_fitness:
                    self.global_best_fitness = new_global_fitness
                    self.global_best_position = refined_global_position

            self.w = self.weight_linear_decrease(generation)
            # Update particles in each sub-swarm
            for swarm in self.swarms:
                for particle in swarm:
                    self.update_velocity(particle, self.global_best_position)
                    self.update_position(particle)

            # Inter-swarm communication: Share best positions
            if generation % 10 == 0:
                for i, swarm in enumerate(self.swarms):
                    for particle in swarm:
                        self.update_velocity(particle, local_best_positions[i])
                        self.update_position(particle)

            # Migration strategy: Allow particles to migrate between sub-swarms
            if generation % 20 == 0:
                self.migrate_particles()

            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            # Add TensorBoard logging
            self.writer.add_scalar('Fitness/Best', self.global_best_fitness, generation)
            self.writer.add_scalar('Fitness/Average', average_particle_fitness, generation)
            self.writer.add_scalar('Parameters/Inertia_Weight', self.w, generation)
            
            # Log subswarm best fitnesses
            for i, fitness in enumerate(local_best_fitnesses):
                self.writer.add_scalar(f'Fitness/Subswarm_{i+1}_Best', fitness, generation)
            
            # Log best position dimensions
            if generation % 5 == 0:
                for i, pos in enumerate(self.global_best_position):
                    self.writer.add_scalar(f'Best_Position/Dimension_{i}', pos, generation)
                
                # Flush the writer periodically
                self.writer.flush()
                
                self.plot_convergence(self.model_name)
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Subswarm Optimisation - Best Fitness: {self.global_best_fitness:.6e}")

        # Make sure to flush at the end
        self.writer.flush()
        return self.global_best_position, self.global_best_fitness

    def migrate_particles(self):
        # Simple migration strategy: randomly select particles to migrate
        for i in range(self.num_sub_swarms):
            if len(self.swarms[i]) > 1:
                # Select a random particle index to migrate
                particle_index = random.randrange(len(self.swarms[i]))
                # Get the particle to migrate
                particle_to_migrate = self.swarms[i][particle_index]
                # Select a random target sub-swarm
                target_swarm_index = random.choice([j for j in range(self.num_sub_swarms) if j != i])
                # Move the particle to the target sub-swarm
                self.swarms[target_swarm_index].append(particle_to_migrate)
                # Remove the particle from the original swarm using its index
                self.swarms[i].pop(particle_index)

    def plot_convergence(self, model_name):
        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
        
        generations = range(len(self.global_best_fitness_array))

        # Create directory if it doesn't exist
        os.makedirs(f'results/{model_name}/particle_subswarm_optimisation', exist_ok=True)
        
        # Plot overall convergence
        file_path = f'results/{model_name}/particle_subswarm_optimisation/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Plot global best fitness
        plt.plot(generations, self.global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        
        # Plot average particle fitness
        plt.plot(generations, self.average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        
        # Track and plot best fitness for each subswarm
        if hasattr(self, 'subswarm_best_fitnesses') and len(self.subswarm_best_fitnesses) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitnesses):
                if len(subswarm_fitness) > 0:  # Only plot if we have data
                    plt.plot(generations, subswarm_fitness, linewidth=2, 
                             label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_fitnesses') and len(self.subswarm_avg_fitnesses) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_fitnesses):
                if len(subswarm_avg) > 0:  # Only plot if we have data
                    plt.plot(generations, subswarm_avg, linewidth=2, 
                             label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
        
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness (Lower is Better)', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        
        # Only add legend if we have plotted something
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend(fontsize=12, loc='upper right')
        
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Plot last 10 generations
        file_path = f'results/{model_name}/particle_subswarm_optimisation/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Only plot if we have at least 10 generations
        if len(self.global_best_fitness_array) >= 10:
            plt.plot(self.global_best_fitness_array[-10:], linewidth=2, label='Global Best Fitness', color='black')
            plt.plot(self.average_particle_fitness_array[-10:], linewidth=2, label='Overall Average Fitness', color='blue', alpha=0.7)
            
            # Plot last 10 generations for each subswarm
            if hasattr(self, 'subswarm_best_fitnesses') and len(self.subswarm_best_fitnesses) > 0:
                for i, subswarm_fitness in enumerate(self.subswarm_best_fitnesses):
                    if len(subswarm_fitness) >= 10:  # Only plot if we have enough data
                        plt.plot(subswarm_fitness[-10:], linewidth=2, 
                                 label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            
            # Plot last 10 generations of average fitness for each subswarm
            if hasattr(self, 'subswarm_avg_fitnesses') and len(self.subswarm_avg_fitnesses) > 0:
                for i, subswarm_avg in enumerate(self.subswarm_avg_fitnesses):
                    if len(subswarm_avg) >= 10:  # Only plot if we have enough data
                        plt.plot(subswarm_avg[-10:], linewidth=2, 
                                 label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
            
            plt.xlabel('Generations', fontsize=16)
            plt.ylabel('Fitness (Lower is Better)', fontsize=16)
            plt.title('Particle SubSwarm Optimisation (Last 10 Generations)', fontsize=18)
            
            # Only add legend if we have plotted something
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(handles) > 0:
                plt.legend(fontsize=12, loc='upper right')
            
            plt.grid(True)
            plt.savefig(file_path)
        
        plt.close()