import os
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time  # Add this import
import multiprocessing
from multiprocessing import Pool, cpu_count
import concurrent.futures
import json
from datetime import datetime

from src.envs.pso.env_wrapped_ea import pso_wrapped_env
from src.envs.universal_physics_plotter import universal_physics_plotter

from configs.evolutionary_algorithms_config import subsonic_pso_params, supersonic_pso_params, flip_over_boostbackburn_pso_params, ballistic_arc_descent_pso_params, landing_burn_pure_throttle_pso_params, landing_burn_pso_params

# Define worker function outside of class for multiprocessing
def evaluate_worker_function(args):
    position, flight_phase, enable_wind, stochastic_wind, horiontal_wind_percentile = args
    # Create a fresh model instance for each worker process
    model = pso_wrapped_env(flight_phase=flight_phase,
                          enable_wind=enable_wind,
                          stochastic_wind=stochastic_wind,
                          horiontal_wind_percentile=horiontal_wind_percentile)
    fitness = model.objective_function(position)
    model.reset()
    return fitness

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
                #self.model.plot_results(self.global_best_position, self.plots_dir)
                # Commented out to save time for now.
                self.plot_convergence()
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Swarm Optimisation - Best Fitness: {self.global_best_fitness:.4e}")

            # Save the swarm state every 50 generations
            if generation % 50 == 0:
                self.save_swarm(f"swarm_state_gen_{generation}.pkl")

        return self.global_best_position, self.global_best_fitness

    def plot_convergence(self):
        # Is inverse fitness so we gotta flip it...

        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
        generations = range(len(self.global_best_fitness_array))

        # Create directory if it doesn't exist
        plot_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Convert lists to numpy arrays for element-wise negation
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        # Plot overall convergence
        file_path = f'{plot_dir}/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Plot global best fitness
        plt.plot(generations, -global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        
        # Plot average particle fitness
        plt.plot(generations, -average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        
        # Track and plot best fitness for each subswarm
        if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                subswarm_fitness_array = np.array(subswarm_fitness)
                plt.plot(generations, -subswarm_fitness_array, linewidth=2, 
                         label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_array') and len(self.subswarm_avg_array) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_array):
                if len(subswarm_avg) > 0:  # Only plot if we have data
                    subswarm_avg_array = np.array(subswarm_avg)
                    plt.plot(generations, -subswarm_avg_array, linewidth=2, 
                             label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
        
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        
        # Only add legend if we have plotted something
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend(fontsize=12, loc='lower right')
        
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Plot last 10 generations
        last_10_generations_idx = generations[-10:]
        file_path = f'{plot_dir}/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Only plot if we have at least 10 generations
        if len(self.global_best_fitness_array) >= 10:
            last_10_best_fitness = np.array(self.global_best_fitness_array[-10:])
            plt.plot(last_10_generations_idx, -last_10_best_fitness, linewidth=2, label='Global Best Fitness', color='black')
            
            # Plot last 10 generations for each subswarm
            if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
                for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                    if len(subswarm_fitness) >= 10:  # Only plot if we have enough data
                        last_10_subswarm_fitness = np.array(subswarm_fitness[-10:])
                        plt.plot(last_10_generations_idx, -last_10_subswarm_fitness, linewidth=2, 
                                 label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            
            plt.xlabel('Generations', fontsize=16)
            plt.ylabel('Fitness', fontsize=16)
            plt.title('Particle SubSwarm Optimisation (Last 10 Generations)', fontsize=18)
            
            # Only add legend if we have plotted something
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(handles) > 0:
                plt.legend(fontsize=12, loc='lower right')
            
            plt.grid(True)
            plt.savefig(file_path)        
        plt.close()
        
        # Create subplot figure with best fitness on top and average on bottom
        self.plot_convergence_subplots()

    def plot_convergence_subplots(self):
        """Create a 2-subplot figure with best fitness on top and average fitness on bottom."""
        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
            
        generations = range(len(self.global_best_fitness_array))
        plot_dir = f'{self.base_save_dir}/plots'
        
        # Convert lists to numpy arrays for element-wise negation (fitness is inverted)
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
        plt.rcParams.update({'font.size': 14})
        
        # Top subplot - Best Fitness
        ax1.plot(generations, -global_best_fitness_array, linewidth=2.5, 
                 label='Global Best Fitness', color='black')
        
        # Plot best fitness for each subswarm
        if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                if len(subswarm_fitness) > 0:  # Only plot if we have data
                    subswarm_fitness_array = np.array(subswarm_fitness)
                    ax1.plot(generations, -subswarm_fitness_array, linewidth=2, 
                             label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        ax1.set_ylabel('Best Fitness', fontsize=16)
        ax1.set_title('Particle SubSwarm Optimisation - Best Fitness per Subswarm', fontsize=18)
        ax1.grid(True)
        ax1.legend(fontsize=12, loc='lower right')
        
        # Bottom subplot - Average Fitness
        ax2.plot(generations, -average_particle_fitness_array, linewidth=2.5, 
                 label='Overall Average Fitness', color='blue')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_array') and len(self.subswarm_avg_array) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_array):
                if len(subswarm_avg) > 0:  # Only plot if we have data
                    subswarm_avg_array = np.array(subswarm_avg)
                    ax2.plot(generations, -subswarm_avg_array, linewidth=2, 
                             label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle='--')
        
        ax2.set_xlabel('Generations', fontsize=16)
        ax2.set_ylabel('Average Fitness', fontsize=16)
        ax2.set_title('Particle SubSwarm Optimisation - Average Fitness per Subswarm', fontsize=18)
        ax2.grid(True)
        ax2.legend(fontsize=12, loc='lower right')
        
        # Adjust layout and save
        plt.tight_layout()
        file_path = f'{plot_dir}/convergence_subplots.png'
        fig.savefig(file_path)
        plt.close(fig)

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
                 horiontal_wind_percentile = 50,
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

        # Store these attributes explicitly for use in parallel_evaluate
        self.enable_wind = enable_wind
        self.stochastic_wind = stochastic_wind
        self.horiontal_wind_percentile = horiontal_wind_percentile
        self.flight_phase = flight_phase

        # Multiprocessing configuration
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = num_processes if num_processes else cpu_count()
        
        # Save interval
        self.save_interval = save_interval
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create base save directory with timestamp
        self.base_save_dir = f'data/pso_saves/{self.flight_phase}/run_{self.timestamp}'
        os.makedirs(self.base_save_dir, exist_ok=True)
        
        # Create subdirectories
        self.saves_dir = f'{self.base_save_dir}/saves'
        self.metrics_dir = f'{self.base_save_dir}/metrics'
        self.trajectory_dir = f'{self.base_save_dir}/trajectory_data'
        self.plots_dir = f'{self.base_save_dir}/plots'
        
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        # Set swarm pickle path
        self.save_swarm_dir = f'{self.saves_dir}/swarm.pkl'

        # For writing best individual to csv periodically
        self.individual_dictionary_initial = self.model.mock_dictionary_of_opt_params

        # Save configuration parameters to JSON at initialization only
        self.save_config_to_json()

        if load_swarms:
            self.load_swarms()
        
    def parallel_evaluate(self, positions):
        results = []
        
        if self.use_multiprocessing:
            # Prepare arguments for the worker function
            args_list = [(position, self.flight_phase, self.enable_wind, 
                         self.stochastic_wind, self.horiontal_wind_percentile) 
                         for position in positions]
            
            # Use Pool instead of ProcessPoolExecutor for better compatibility
            with Pool(processes=self.num_processes) as pool:
                results = pool.map(evaluate_worker_function, args_list)
        else:
            # Sequential evaluation
            for position in positions:
                fitness = self.model.objective_function(position)
                self.model.reset()
                results.append(fitness)
            
        return results
            
    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.best_fitness_array_array = []
        self.best_individual_array_array = []
        self.initialize_swarms()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.subswarm_best_positions = []
        self.subswarm_best_fitnesses = []
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]

    def __call__(self):
        self.run()

    def __del__(self):
        pass

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
        
        # Create directories for saving generation-by-generation metrics
        os.makedirs(f'data/pso_saves/{self.flight_phase}/metrics/', exist_ok=True)
        
        for generation in pbar:
            start_time = time.time()  # Start time for the generation
            
            # Dictionary to store per-swarm metrics for this generation
            generation_metrics = {
                'generation': generation,
                'swarm_metrics': []
            }

            all_particle_fitnesses = []
            
            # Evaluate each sub-swarm independently
            for swarm_idx, swarm in enumerate(self.swarms):
                all_particles_swarm_idx_fitnesses = []
                
                if self.use_multiprocessing:
                    # Extract positions for parallel evaluation
                    positions = [p['position'] for p in swarm]
                    
                    # Use ThreadPoolExecutor for parallel evaluation (avoids pickling issues)
                    fitnesses = self.parallel_evaluate(positions)
                    
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

                # Calculate swarm metrics
                swarm_best_fitness = self.subswarm_best_fitnesses[swarm_idx]
                swarm_avg_fitness = np.mean(all_particles_swarm_idx_fitnesses)
                swarm_min_fitness = np.min(all_particles_swarm_idx_fitnesses)
                swarm_max_fitness = np.max(all_particles_swarm_idx_fitnesses)
                swarm_std_fitness = np.std(all_particles_swarm_idx_fitnesses)
                
                # Store metrics for this swarm
                swarm_metrics = {
                    'swarm_idx': swarm_idx,
                    'best_fitness': swarm_best_fitness,
                    'avg_fitness': swarm_avg_fitness,
                    'min_fitness': swarm_min_fitness,
                    'max_fitness': swarm_max_fitness,
                    'std_fitness': swarm_std_fitness,
                    'num_particles': len(swarm)
                }
                generation_metrics['swarm_metrics'].append(swarm_metrics)

                # Update subswarm best fitness and position arrays
                self.subswarm_best_fitness_array[swarm_idx].append(self.subswarm_best_fitnesses[swarm_idx])
                self.subswarm_best_position_array[swarm_idx].append(self.subswarm_best_positions[swarm_idx])

                # Update subswarm average fitness array
                self.subswarm_avg_array[swarm_idx].append(np.mean(all_particles_swarm_idx_fitnesses))
                
            # Update global best from subswarm bests
            for i, fitness in enumerate(self.subswarm_best_fitnesses):
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.subswarm_best_positions[i].copy()

            # Calculate average fitness across all particles
            average_particle_fitness = np.mean(all_particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)
            
            # Add global metrics to generation_metrics
            generation_metrics['global_best_fitness'] = self.global_best_fitness
            generation_metrics['global_avg_fitness'] = average_particle_fitness
            
            # Save generation metrics
            self.save_generation_metrics(generation_metrics, generation)

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
            
            # Log best position dimensions and plot periodically
            if generation % self.save_interval == 0 and generation != 0:
                self.plot_convergence()
                # Commented out to save time for now.
                self.model.plot_results(self.global_best_position, self.plots_dir + '/')
                
                # Save trajectory data for the best individual
                trajectory_data = self.collect_trajectory_data(self.global_best_position)
                self.save_trajectory_data(trajectory_data)

                self.save()
                self.save_results()
            if generation == self.re_initialise_generation:
                self.re_initialise_swarms()
            
            # Update tqdm description with best fitness
            pbar.set_description(f"Particle Subswarm Optimisation - Best Fitness: {self.global_best_fitness:.6e}")
        
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
        # Is inverse fitness so we gotta flip it...

        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
        generations = range(len(self.global_best_fitness_array))

        # Create directory if it doesn't exist
        plot_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Convert lists to numpy arrays for element-wise negation
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        # Plot overall convergence
        file_path = f'{plot_dir}/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Plot global best fitness
        plt.plot(generations, -global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        
        # Plot average particle fitness
        plt.plot(generations, -average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        
        # Track and plot best fitness for each subswarm
        if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                subswarm_fitness_array = np.array(subswarm_fitness)
                plt.plot(generations, -subswarm_fitness_array, linewidth=2, 
                         label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_array') and len(self.subswarm_avg_array) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_array):
                if len(subswarm_avg) > 0:  # Only plot if we have data
                    subswarm_avg_array = np.array(subswarm_avg)
                    plt.plot(generations, -subswarm_avg_array, linewidth=2, 
                             label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
        
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        
        # Only add legend if we have plotted something
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend(fontsize=12, loc='lower right')
        
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Plot last 10 generations
        last_10_generations_idx = generations[-10:]
        file_path = f'{plot_dir}/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Only plot if we have at least 10 generations
        if len(self.global_best_fitness_array) >= 10:
            last_10_best_fitness = np.array(self.global_best_fitness_array[-10:])
            plt.plot(last_10_generations_idx, -last_10_best_fitness, linewidth=2, label='Global Best Fitness', color='black')
            
            # Plot last 10 generations for each subswarm
            if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
                for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                    if len(subswarm_fitness) >= 10:  # Only plot if we have enough data
                        last_10_subswarm_fitness = np.array(subswarm_fitness[-10:])
                        plt.plot(last_10_generations_idx, -last_10_subswarm_fitness, linewidth=2, 
                                 label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            
            plt.xlabel('Generations', fontsize=16)
            plt.ylabel('Fitness', fontsize=16)
            plt.title('Particle SubSwarm Optimisation (Last 10 Generations)', fontsize=18)
            
            # Only add legend if we have plotted something
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(handles) > 0:
                plt.legend(fontsize=12, loc='lower right')
            
            plt.grid(True)
            plt.savefig(file_path)        
        plt.close()
        
        # Create subplot figure with best fitness on top and average on bottom
        self.plot_convergence_subplots()

    def plot_convergence_subplots(self):
        """Create a 2-subplot figure with best fitness on top and average fitness on bottom."""
        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
            
        generations = range(len(self.global_best_fitness_array))
        plot_dir = f'{self.base_save_dir}/plots'
        
        # Convert lists to numpy arrays for element-wise negation (fitness is inverted)
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
        plt.rcParams.update({'font.size': 14})
        
        # Top subplot - Best Fitness
        ax1.plot(generations, -global_best_fitness_array, linewidth=2.5, 
                 label='Global Best Fitness', color='black')
        
        # Plot best fitness for each subswarm
        if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                if len(subswarm_fitness) > 0:  # Only plot if we have data
                    subswarm_fitness_array = np.array(subswarm_fitness)
                    ax1.plot(generations, -subswarm_fitness_array, linewidth=2, 
                             label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        ax1.set_ylabel('Best Fitness', fontsize=16)
        ax1.set_title('Particle SubSwarm Optimisation - Best Fitness per Subswarm', fontsize=18)
        ax1.grid(True)
        ax1.legend(fontsize=12, loc='lower right')
        
        # Bottom subplot - Average Fitness
        ax2.plot(generations, -average_particle_fitness_array, linewidth=2.5, 
                 label='Overall Average Fitness', color='blue')
        
        # Plot average fitness for each subswarm
        if hasattr(self, 'subswarm_avg_array') and len(self.subswarm_avg_array) > 0:
            for i, subswarm_avg in enumerate(self.subswarm_avg_array):
                if len(subswarm_avg) > 0:  # Only plot if we have data
                    subswarm_avg_array = np.array(subswarm_avg)
                    ax2.plot(generations, -subswarm_avg_array, linewidth=2, 
                             label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle='--')
        
        ax2.set_xlabel('Generations', fontsize=16)
        ax2.set_ylabel('Average Fitness', fontsize=16)
        ax2.set_title('Particle SubSwarm Optimisation - Average Fitness per Subswarm', fontsize=18)
        ax2.grid(True)
        ax2.legend(fontsize=12, loc='lower right')
        
        # Adjust layout and save
        plt.tight_layout()
        file_path = f'{plot_dir}/convergence_subplots.png'
        fig.savefig(file_path)
        plt.close(fig)

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
            
        except FileNotFoundError:
            print(f"No swarm state file found at {file_path}. Starting with fresh swarms.")

    def save_results(self):
        # Save results to CSV
        file_path = f'{self.base_save_dir}/particle_subswarm_optimisation_results.csv'
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

        # Save fitness history to CSV
        self.save_fitness_history()
        
    def save_fitness_history(self):
        """Save the fitness history and generations to a CSV file."""
        # Prepare data for CSV
        generations = list(range(len(self.global_best_fitness_array)))
        
        # Create DataFrame with global fitness data
        data = {
            'Generation': generations,
            'Global_Best_Fitness': self.global_best_fitness_array,
            'Average_Fitness': self.average_particle_fitness_array
        }
        
        # Add subswarm data
        for i in range(self.num_sub_swarms):
            if len(self.subswarm_best_fitness_array[i]) > 0:
                # Pad arrays if needed to match generations length
                if len(self.subswarm_best_fitness_array[i]) < len(generations):
                    pad_length = len(generations) - len(self.subswarm_best_fitness_array[i])
                    padded_array = self.subswarm_best_fitness_array[i] + [None] * pad_length
                else:
                    padded_array = self.subswarm_best_fitness_array[i][:len(generations)]
                data[f'Subswarm_{i+1}_Best'] = padded_array
            
            if len(self.subswarm_avg_array[i]) > 0:
                # Pad arrays if needed
                if len(self.subswarm_avg_array[i]) < len(generations):
                    pad_length = len(generations) - len(self.subswarm_avg_array[i])
                    padded_array = self.subswarm_avg_array[i] + [None] * pad_length
                else:
                    padded_array = self.subswarm_avg_array[i][:len(generations)]
                data[f'Subswarm_{i+1}_Average'] = padded_array
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        file_path = f'{self.metrics_dir}/fitness_history.csv'
        df.to_csv(file_path, index=False)
        print(f"Fitness history saved to {file_path}")        

    def plot_results(self, individual):
        # Create plots directory if it doesn't exist
        plots_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run the simulation and collect trajectory data
        trajectory_data = self.collect_trajectory_data(individual)
        
        # Save the trajectory data
        self.save_trajectory_data(trajectory_data)
        
        # Apply the individual to the model
        self.model.individual_update_model(individual)
        
        # Call universal_physics_plotter directly with our plots directory
        # Commented out to save time for now.
        #universal_physics_plotter(self.model.env,
        #                         self.model.actor,
        #                         plots_dir + '/',
        #                         flight_phase=self.flight_phase,
        #                         type='pso')

    def collect_trajectory_data(self, individual):
        """Collect trajectory data by running a simulation with the given individual."""
        self.model.individual_update_model(individual)
        state = self.model.env.reset()

        trajectory_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'info': []
        }

        done_or_truncated = False
        while not done_or_truncated:
            action = self.model.actor.forward(state)
            next_state, reward, done, truncated, info = self.model.env.step(action)
            
            # Store data
            trajectory_data['states'].append(state.tolist() if hasattr(state, 'tolist') else state)
            trajectory_data['actions'].append(action.detach().numpy().tolist() if hasattr(action, 'detach') else action)
            trajectory_data['rewards'].append(reward)
            trajectory_data['info'].append(info)
            
            # Update for next iteration
            done_or_truncated = done or truncated
            state = next_state
            
        return trajectory_data

    def save_trajectory_data(self, trajectory_data):
        """Save the trajectory data to a CSV file."""        
        # Create directory if it doesn't exist (though it should already exist)
        os.makedirs(self.trajectory_dir, exist_ok=True)
        
        # Save states and actions to CSV
        states_df = pd.DataFrame(trajectory_data['states'])
        actions_df = pd.DataFrame(trajectory_data['actions'])
        rewards_df = pd.DataFrame(trajectory_data['rewards'], columns=['reward'])
        
        # Save state-action data
        states_df.to_csv(f'{self.trajectory_dir}/states.csv', index=False)
        actions_df.to_csv(f'{self.trajectory_dir}/actions.csv', index=False)
        rewards_df.to_csv(f'{self.trajectory_dir}/rewards.csv', index=False)
        
        # Extract and save info data - much simpler approach
        if trajectory_data['info']:
            # Create flattened dictionary from all nested info dictionaries
            flat_data = []
            
            for info in trajectory_data['info']:
                flat_info = {}
                
                # Function to recursively flatten nested dictionaries
                def flatten_dict(d, prefix=''):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            flatten_dict(value, f"{prefix}{key}_")
                        else:
                            flat_info[f"{prefix}{key}"] = value
                
                # Flatten the entire info dictionary
                flatten_dict(info)
                flat_data.append(flat_info)
            
            # Create DataFrame from flattened data
            info_df = pd.DataFrame(flat_data)
            info_df.to_csv(f'{self.trajectory_dir}/info_data.csv', index=False)

    def save_generation_metrics(self, metrics, generation):
        """Save detailed metrics for each generation"""
        # Create separate DataFrames for each subswarm
        for swarm_metric in metrics['swarm_metrics']:
            swarm_idx = swarm_metric['swarm_idx']
            # Create a DataFrame with just this swarm's metrics
            swarm_df = pd.DataFrame([swarm_metric])
            swarm_df['generation'] = generation
            swarm_df['global_best_fitness'] = metrics['global_best_fitness']
            swarm_df['global_avg_fitness'] = metrics['global_avg_fitness']
            
            # Save to a CSV file for this specific subswarm
            swarm_file = f'{self.metrics_dir}/subswarm_{swarm_idx}_metrics.csv'
            
            # If it's the first generation, create a new file with headers
            if generation == 0:
                swarm_df.to_csv(swarm_file, index=False, mode='w')
            else:
                # Append to the existing file without headers
                swarm_df.to_csv(swarm_file, index=False, mode='a', header=False)
        
        # Also save the global metrics
        global_df = pd.DataFrame({
            'generation': [generation],
            'global_best_fitness': [metrics['global_best_fitness']],
            'global_avg_fitness': [metrics['global_avg_fitness']]
        })
        
        # Save global metrics to a separate file
        global_file = f'{self.metrics_dir}/global_metrics.csv'
        if generation == 0:
            global_df.to_csv(global_file, index=False, mode='w')
        else:
            global_df.to_csv(global_file, index=False, mode='a', header=False)

    def save_config_to_json(self):
        """Save all configuration parameters to a JSON file."""
        # Just use the imported parameter dictionary directly
        config = self.pso_params.copy()
        
        # Add a few runtime parameters that aren't in the original config
        config.update({
            'flight_phase': self.flight_phase,
            'enable_wind': self.enable_wind,
            'stochastic_wind': self.stochastic_wind,
            'horiontal_wind_percentile': self.horiontal_wind_percentile,
            'use_multiprocessing': self.use_multiprocessing,
            'num_processes': self.num_processes,
            'save_interval': self.save_interval,
            'timestamp': self.timestamp
        })
        
        # Save to JSON file
        with open(f'{self.base_save_dir}/pso_config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Configuration saved to {self.base_save_dir}/pso_config.json")        