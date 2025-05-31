import os
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time  # Add this import
import multiprocessing
from multiprocessing import Pool, cpu_count

from src.envs.pso.env_wrapped_ea import pso_wrapped_env

from configs.evolutionary_algorithms_config import subsonic_pso_params, supersonic_pso_params, flip_over_boostbackburn_pso_params, ballistic_arc_descent_pso_params, landing_burn_pure_throttle_pso_params, landing_burn_pso_params

class ParticleSwarmOptimisation:
    def __init__(self, flight_phase, enable_wind = False, stochastic_wind = False, horiontal_wind_percentile = 95):
        if flight_phase == 'subsonic':
            self.pso_params = subsonic_pso_params
        elif flight_phase == 'supersonic':
            self.pso_params = supersonic_pso_params
        elif flight_phase == 'flip_over_boostbackburn':
            self.pso_params = flip_over_boostbackburn_pso_params
        elif flight_phase == 'ballistic_arc_descent':
            self.pso_params = ballistic_arc_descent_pso_params
        elif flight_phase == 'landing_burn_pure_throttle':
            self.pso_params = landing_burn_pure_throttle_pso_params
        elif flight_phase == 'landing_burn':
            self.pso_params = landing_burn_pso_params

        self.model = pso_wrapped_env(flight_phase, enable_wind = enable_wind, stochastic_wind = stochastic_wind, horiontal_wind_percentile = horiontal_wind_percentile)

        self.pop_size = self.pso_params['pop_size']
        self.generations = self.pso_params['generations']
        self.w_start = self.pso_params['w_start']
        self.w_end = self.pso_params['w_end']
        self.c1 = self.pso_params['c1']
        self.c2 = self.pso_params['c2']

        self.bounds = self.model.bounds
        self.flight_phase = flight_phase

        self.best_fitness_array = []
        self.best_individual_array = []

        self.initialize_swarm()   
        self.w = self.w_start     

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []
        self.average_particle_fitness_array = []

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
            particle_fitnesses = []
            for i, particle in enumerate(self.swarm):
                fitness = self.evaluate_particle(particle)
                particle_fitnesses.append(fitness)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()

            average_particle_fitness = np.mean(particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)            

            self.w = self.weight_linear_decrease(generation)
            for i, particle in enumerate(self.swarm):
                self.update_velocity(particle, self.global_best_position)
                self.update_position(particle)

            
 
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            if generation % 5 == 0 and generation != 0:
                self.model.plot_results(self.global_best_position)

                self.plot_convergence()
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Swarm Optimisation - Best Fitness: {self.global_best_fitness:.4e}")

            # Save the swarm state every 50 generations
            if generation % 50 == 0:
                self.save_swarm(f"swarm_state_gen_{generation}.pkl")

        return self.global_best_position, self.global_best_fitness

    def plot_convergence(self):
        generations = range(len(self.global_best_fitness_array))

        file_path = f'results/particle_swarm_optimisation/{self.flight_phase}/convergence.png'

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

        file_path = f'results/particle_swarm_optimisation/{self.flight_phase}/last_10_fitnesses.png'
        last_10_generations_idx = generations[-10:]
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(last_10_generations_idx, self.global_best_fitness_array[-10:], linewidth=2, label='Best Fitness')
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle Swarm Optimisation Convergence', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def save_swarm(self, file_path):
        """Save the current state of the swarm to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.swarm, f)
        print(f"Swarm state saved to {file_path}")

    def load_swarm(self, file_path):
        """Load the swarm state from a file."""
        with open(file_path, 'rb') as f:
            self.swarm = pickle.load(f)
        print(f"Swarm state loaded from {file_path}")

        # Update global best based on loaded swarm
        for particle in self.swarm:
            if particle['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = particle['best_fitness']
                self.global_best_position = particle['best_position'].copy()



class ParticleSubswarmOptimisation(ParticleSwarmOptimisation):
    def __init__(self,
                 flight_phase,
                 save_interval,
                 enable_wind = False,
                 stochastic_wind = False,
                 horiontal_wind_percentile = 95,
                 load_swarms = False,
                 use_multiprocessing = True,
                 num_processes = None):
        super().__init__(flight_phase, enable_wind = enable_wind, stochastic_wind = stochastic_wind, horiontal_wind_percentile = horiontal_wind_percentile)
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle']
        self.num_sub_swarms = self.pso_params["num_sub_swarms"]
        self.communication_freq = self.pso_params.get("communication_freq", 10)
        self.migration_freq = self.pso_params.get("migration_freq", 20)
        self.number_of_migrants = self.pso_params.get("number_of_migrants", 1)
        self.re_initialise_number_of_particles = self.pso_params.get("re_initialise_number_of_particles", 500)
        self.re_initialise_generation = self.pso_params.get("re_initialise_generation", 60)
        self.initialize_swarms()

        # Multiprocessing configuration
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = num_processes if num_processes else cpu_count()
        
        # Save interval
        self.save_interval = save_interval
        
        # Use a single log directory for all subswarm runs of this model
        base_log_dir = f"data/pso_saves/{self.flight_phase}/runs/"
        self.writer = SummaryWriter(log_dir=base_log_dir)

        # Make pickle dump directory
        self.save_swarm_dir = f'data/pso_saves/{self.flight_phase}/saves/swarm.pkl'

        # For writing best individual to csv periodically
        self.individual_dictionary_initial = self.model.mock_dictionary_of_opt_params

        if load_swarms:
            self.load_swarms()
        
    # Helper function for multiprocessing
    def _evaluate_particle_wrapper(self, particle_data):
        """Wrapper function for particle evaluation in multiprocessing"""
        position = particle_data['position']
        fitness = self.model.objective_function(position)
        self.model.reset()
        return fitness
        
    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.best_fitness_array_array = []
        self.best_individual_array_array = []
        self.initialize_swarms()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Close the previous writer and create a new one with the same log directory
        self.writer.close()
        self.writer = SummaryWriter(log_dir=self.writer.log_dir)

        self.subswarm_best_positions = []
        self.subswarm_best_fitnesses = []
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]

    def __call__(self):
        self.run()

    def __del__(self):
        """Ensure the writer is closed when the object is deleted"""
        if hasattr(self, 'writer'):
            self.writer.close()

    def re_initialise_swarms(self):
        '''
        The purpose of this function is to re-initialise swarms at a set generation,
        such that the problem can be ran in a feasible time.
        The swarms will select the best performing particles from the previous generation,
        and then re-initialise the swarm with these particles.
        '''
        number_of_particle_per_swarm_new = self.re_initialise_number_of_particles // self.num_sub_swarms
        for swarm_idx, swarm in enumerate(self.swarms):
            # Select the best performing particles from the previous generation
            best_particles = sorted(swarm, key=lambda x: x['best_fitness'])[:number_of_particle_per_swarm_new]
            self.swarms[swarm_idx] = best_particles
        print(f'Swarms re-initialised to {number_of_particle_per_swarm_new} particles each')


    def initialize_swarms(self):
        self.swarms = []
        sub_swarm_size = self.pop_size // self.num_sub_swarms
        
        # Each subswarm will have its own local best
        self.subswarm_best_positions = []
        self.subswarm_best_fitnesses = []
        # List of lists with inner lists number equal to number of subswarms
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]
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
            
            # Initialize each subswarm's best position and fitness
            self.subswarm_best_positions.append(None)
            self.subswarm_best_fitnesses.append(float('inf'))

    def run(self):
        pbar = tqdm(range(self.generations), desc='Particle Swarm Optimisation with Subswarms')
        
        # Initialize multiprocessing pool if enabled
        if self.use_multiprocessing:
            # Use 'spawn' method for better compatibility
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(processes=self.num_processes)
            print(f"Using multiprocessing with {self.num_processes} processes")
        
        for generation in pbar:
            start_time = time.time()  # Start time for the generation

            all_particle_fitnesses = []
            
            # Evaluate each sub-swarm independently
            for swarm_idx, swarm in enumerate(self.swarms):
                all_particles_swarm_idx_fitnesses = []
                
                if self.use_multiprocessing:
                    # Prepare data for parallel processing
                    particle_positions = [{'position': p['position']} for p in swarm]
                    
                    # Parallel evaluation of particles
                    fitnesses = pool.map(self._evaluate_particle_wrapper, particle_positions)
                    
                    # Update particles with their fitness values
                    for i, (particle, fitness) in enumerate(zip(swarm, fitnesses)):
                        all_particle_fitnesses.append(fitness)
                        all_particles_swarm_idx_fitnesses.append(fitness)
                        
                        # Update particle's personal best
                        if fitness < particle['best_fitness']:
                            particle['best_fitness'] = fitness
                            particle['best_position'] = particle['position'].copy()
                        
                        # Update subswarm's best
                        if fitness < self.subswarm_best_fitnesses[swarm_idx]:
                            self.subswarm_best_fitnesses[swarm_idx] = fitness
                            self.subswarm_best_positions[swarm_idx] = particle['position'].copy()
                else:
                    # Original sequential evaluation
                    for particle in swarm:
                        fitness = self.evaluate_particle(particle)
                        all_particle_fitnesses.append(fitness)
                        all_particles_swarm_idx_fitnesses.append(fitness)
                        
                        # Update particle's personal best
                        if fitness < particle['best_fitness']:
                            particle['best_fitness'] = fitness
                            particle['best_position'] = particle['position'].copy()
                        
                        # Update subswarm's best
                        if fitness < self.subswarm_best_fitnesses[swarm_idx]:
                            self.subswarm_best_fitnesses[swarm_idx] = fitness
                            self.subswarm_best_positions[swarm_idx] = particle['position'].copy()

                # Update subswarm best fitness and position arrays
                self.subswarm_best_fitness_array[swarm_idx].append(self.subswarm_best_fitnesses[swarm_idx])
                self.subswarm_best_position_array[swarm_idx].append(self.subswarm_best_positions[swarm_idx])

                # Update subswarm average fitness array
                self.subswarm_avg_array[swarm_idx].append(np.mean(all_particles_swarm_idx_fitnesses))

                # Log histogram of best fitnesses for this subswarm
                self.writer.add_histogram(f'Subswarm_{swarm_idx+1}/Fitnesses', 
                                          np.array(all_particles_swarm_idx_fitnesses), 
                                          generation)
                
            # Log histogram of best fitnesses for all subswarms
            self.writer.add_histogram('All_Subswarms/Fitnesses', 
                                      np.array(all_particle_fitnesses), 
                                      generation)
            
            # Log histogram of velocities for all subswarms
            velocities = np.array([particle['velocity'] for swarm in self.swarms for particle in swarm])
            for dim in range(velocities.shape[1]):
                self.writer.add_histogram(f'All_Subswarms/Velocities/Dimension_{dim}', 
                                            velocities[:, dim], 
                                            generation)

            # Update global best from subswarm bests
            for i, fitness in enumerate(self.subswarm_best_fitnesses):
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.subswarm_best_positions[i].copy()

            # Calculate average fitness across all particles
            average_particle_fitness = np.mean(all_particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)

            # Measure and log the time taken for this generation
            end_time = time.time()
            generation_time = end_time - start_time
            self.writer.add_scalar('Time/Generation', generation_time, generation)

            # Update inertia weight
            self.w = self.weight_linear_decrease(generation)
            
            # Update particles in each sub-swarm using their OWN subswarm best
            for swarm_idx, swarm in enumerate(self.swarms):
                for particle in swarm:
                    self.update_velocity_with_local_best(
                        particle, 
                        self.subswarm_best_positions[swarm_idx]
                    )
                    self.update_position(particle)

            # Inter-swarm communication: Share information periodically
            if generation % self.communication_freq == 0 and generation > 0:
                self.share_information()

            # Migration strategy: Allow particles to migrate between sub-swarms
            if generation % self.migration_freq == 0 and generation > 0:
                self.migrate_particles()

            # Record history
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            # Add TensorBoard logging
            self.writer.add_scalar('Fitness/Global_Best', self.global_best_fitness, generation)
            self.writer.add_scalar('Fitness/Average', average_particle_fitness, generation)
            self.writer.add_scalar('Parameters/Inertia_Weight', self.w, generation)
            
            # Log particle positions as histograms (overall)
            for dim in range(len(self.bounds)):
                all_positions = []
                for swarm in self.swarms:
                    all_positions.extend([p['position'][dim] for p in swarm])
                self.writer.add_histogram(f'All_Particles/Dimension_{dim}', 
                                        np.array(all_positions), 
                                        generation)
            
            # Log best position dimensions and plot periodically
            if generation % self.save_interval == 0 and generation != 0:
                for i, pos in enumerate(self.global_best_position):
                    self.writer.add_scalar(f'Best_Position/Dimension_{i}', pos, generation)
                
                # Flush the writer periodically
                self.writer.flush()
                
                self.plot_convergence()
                self.model.plot_results(self.global_best_position)

                self.save()
                self.save_results()
            if generation == self.re_initialise_generation:
                self.re_initialise_swarms()
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Subswarm Optimisation - Best Fitness: {self.global_best_fitness:.6e}")

        # Clean up the multiprocessing pool if used
        if self.use_multiprocessing:
            pool.close()
            pool.join()
        
        # Make sure to flush at the end
        self.writer.flush()
        
        return self.global_best_position, self.global_best_fitness

    def update_velocity_with_local_best(self, particle, local_best_position):
        """Update velocity using the subswarm's best position instead of global best"""
        inertia = self.w * particle['velocity']
        cognitive = self.c1 * np.random.rand() * (particle['best_position'] - particle['position'])
        social = self.c2 * np.random.rand() * (local_best_position - particle['position'])
        particle['velocity'] = inertia + cognitive + social

    def share_information(self):
        """Share information between subswarms"""
        # Find the best subswarm
        best_swarm_idx = np.argmin(self.subswarm_best_fitnesses)
        best_swarm_position = self.subswarm_best_positions[best_swarm_idx]
        
        # Share the best position with a probability
        for i in range(self.num_sub_swarms):
            if i != best_swarm_idx and random.random() < 0.5:  # 50% chance to share
                # Influence the subswarm's best position slightly
                influence_factor = 0.3  # How much influence the best swarm has
                self.subswarm_best_positions[i] = (
                    (1 - influence_factor) * self.subswarm_best_positions[i] + 
                    influence_factor * best_swarm_position
                )
                
                # Re-evaluate the new position
                temp_particle = {
                    'position': self.subswarm_best_positions[i],
                    'velocity': np.zeros(len(self.bounds)),
                    'best_position': None,
                    'best_fitness': float('inf')
                }
                new_fitness = self.evaluate_particle(temp_particle)
                
                # Update if better
                if new_fitness < self.subswarm_best_fitnesses[i]:
                    self.subswarm_best_fitnesses[i] = new_fitness

    def migrate_particles(self):
        """Allow particles to migrate between subswarms"""
        for i in range(self.num_sub_swarms):
            if len(self.swarms[i]) > 1:
                for _ in range(self.number_of_migrants):
                    # Select a random particle to migrate
                    particle_index = random.randrange(len(self.swarms[i]))
                    particle_to_migrate = self.swarms[i][particle_index]
                    
                    # Select a random target subswarm
                    target_swarm_index = random.choice([j for j in range(self.num_sub_swarms) if j != i])
                    
                    # Move the particle to the target subswarm
                    self.swarms[target_swarm_index].append(particle_to_migrate)
                    self.swarms[i].pop(particle_index)

    def plot_convergence(self):
        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
        generations = range(len(self.global_best_fitness_array))

        # Create directory if it doesn't exist
        os.makedirs(f'results/particle_swarm_optimisation/{self.flight_phase}/', exist_ok=True)
        
        # Plot overall convergence
        file_path = f'results/particle_swarm_optimisation/{self.flight_phase}/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Plot global best fitness
        plt.plot(generations, self.global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        
        # Plot average particle fitness
        plt.plot(generations, self.average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        
        # Track and plot best fitness for each subswarm
        

        if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                plt.plot(generations, subswarm_fitness, linewidth=2, 
                         label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_array') and len(self.subswarm_avg_array) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_array):
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
        last_10_generations_idx = generations[-10:]
        file_path = f'results/particle_swarm_optimisation/{self.flight_phase}/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Only plot if we have at least 10 generations
        if len(self.global_best_fitness_array) >= 10:
            plt.plot(last_10_generations_idx, self.global_best_fitness_array[-10:], linewidth=2, label='Global Best Fitness', color='black')
            
            # Plot last 10 generations for each subswarm
            if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
                for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                    if len(subswarm_fitness) >= 10:  # Only plot if we have enough data
                        plt.plot(last_10_generations_idx, subswarm_fitness[-10:], linewidth=2, 
                                 label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            
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

    def save(self):
        """Save the current state of all subswarms to a file."""
        with open(self.save_swarm_dir, 'wb') as f:
            pickle.dump(self.swarms, f)

    def load_swarms(self):
        """Load the subswarm states from a file."""
        #data/pso_saves/landing_burn/saves/swarm.pkl
        file_path = f'data/pso_saves/{self.flight_phase}/saves/swarm.pkl'
        try:
            with open(file_path, 'rb') as f:
                self.swarms = pickle.load(f)
            print(f"Subswarm states loaded from {file_path}")
            
            # Initialize arrays for tracking metrics
            self.global_best_fitness_array = []
            self.global_best_position_array = []
            self.average_particle_fitness_array = []
            self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
            self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
            self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]
            
            # Re-initialize the best positions and fitnesses
            self.global_best_fitness = float('inf')
            self.global_best_position = None
            self.subswarm_best_positions = [None for _ in range(self.num_sub_swarms)]
            self.subswarm_best_fitnesses = [float('inf') for _ in range(self.num_sub_swarms)]
            
            # Update global and subswarm bests based on loaded swarms
            for swarm_idx, swarm in enumerate(self.swarms):
                for particle in swarm:
                    if particle['best_fitness'] < self.subswarm_best_fitnesses[swarm_idx]:
                        self.subswarm_best_fitnesses[swarm_idx] = particle['best_fitness']
                        self.subswarm_best_positions[swarm_idx] = particle['best_position'].copy()
                    if particle['best_fitness'] < self.global_best_fitness:
                        self.global_best_fitness = particle['best_fitness']
                        self.global_best_position = particle['best_position'].copy()
            
            # Add initial entries to tracking arrays
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)
            
            # Initialize subswarm tracking arrays
            for i in range(self.num_sub_swarms):
                self.subswarm_best_fitness_array[i].append(self.subswarm_best_fitnesses[i])
                self.subswarm_best_position_array[i].append(self.subswarm_best_positions[i])
                
                # Calculate average fitness for this subswarm
                avg_fitness = np.mean([p['best_fitness'] for p in self.swarms[i]])
                self.subswarm_avg_array[i].append(avg_fitness)
            
            # Calculate overall average
            all_fitnesses = [p['best_fitness'] for swarm in self.swarms for p in swarm]
            self.average_particle_fitness_array.append(np.mean(all_fitnesses))
            
            # Log initial state to TensorBoard
            self.writer.add_scalar('Fitness/Global_Best', self.global_best_fitness, 0)
            self.writer.add_scalar('Fitness/Average', self.average_particle_fitness_array[0], 0)
            
        except FileNotFoundError:
            print(f"No swarm state file found at {file_path}. Starting with fresh swarms.")

    def save_results(self):
        # Change file extension from txt to csv
        file_path = f'data/pso_saves/{self.flight_phase}/particle_subswarm_optimisation_results.csv'
        try:
            existing_df = pd.read_csv(file_path, index_col=0)
            file_exists = True
        except (FileNotFoundError, pd.errors.EmptyDataError):
            file_exists = False
        
        column_titles = list(self.individual_dictionary_initial.keys()) + ['Best Fitness']

        if not file_exists:
            # Correct the data structure to match the expected shape
            mock_params = [10e10] * (len(column_titles) - 1) 
            data = [['Particle Subswarm Optimisation'] + mock_params + [10e10]]
            df_columns = ['Algorithm'] + column_titles
            existing_df = pd.DataFrame(data, columns=df_columns)
            existing_df.set_index('Algorithm', inplace=True)
            existing_df.to_csv(file_path)

        best_solution = self.global_best_position
        best_value = self.global_best_fitness
        row_data = dict(zip(column_titles[:-1], best_solution))
        row_data[column_titles[-1]] = best_value
        existing_df.loc['Particle Subswarm Optimisation'] = row_data
        existing_df.to_csv(file_path)        