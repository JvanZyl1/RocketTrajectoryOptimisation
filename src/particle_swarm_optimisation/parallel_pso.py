import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import multiprocessing as mp
from functools import partial

from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation

class ParallelParticleSubswarmOptimisation(ParticleSubswarmOptimisation):
    """Parallel implementation of Particle Swarm Optimization using multiple CPUs"""
    
    def __init__(self, 
                 flight_phase,
                 save_interval,
                 enable_wind=False, 
                 num_processes=None):
        """
        Initialize the parallel PSO algorithm.
        
        Args:
            flight_phase: Flight phase to optimize ('subsonic', 'supersonic', etc.)
            save_interval: How often to save results
            enable_wind: Whether to enable wind in the simulation
            num_processes: Number of CPU processes to use. If None, uses all available CPUs.
        """
        super().__init__(flight_phase, save_interval, enable_wind)
        
        # Set number of processes
        self.num_processes = num_processes if num_processes is not None else mp.cpu_count()
        print(f"Using {self.num_processes} CPU processes for parallel PSO")
        
        # Make sure we don't have more processes than particles
        total_particles = self.pop_size
        if self.num_processes > total_particles:
            print(f"Reducing number of processes to {total_particles} (total particles)")
            self.num_processes = total_particles
    
    def _evaluate_particle_batch(self, batch_particles):
        """Evaluate a batch of particles and return their fitness values"""
        results = []
        
        for particle_idx, particle in batch_particles:
            swarm_idx, particle_in_swarm_idx = divmod(particle_idx, len(self.swarms[0]))
            fitness = self.evaluate_particle(particle)
            results.append((swarm_idx, particle_in_swarm_idx, fitness))
            
        return results
    
    def _prepare_batches(self):
        """Prepare batches of particles for parallel evaluation"""
        batches = []
        particle_idx = 0
        
        # Create list of all particles with their global indices
        all_particles = []
        for swarm_idx, swarm in enumerate(self.swarms):
            for p_idx, particle in enumerate(swarm):
                all_particles.append((particle_idx, particle))
                particle_idx += 1
        
        # Distribute particles evenly across processes
        batch_size = len(all_particles) // self.num_processes
        remainder = len(all_particles) % self.num_processes
        
        start_idx = 0
        for i in range(self.num_processes):
            # Add one extra particle to some batches if we can't divide evenly
            current_batch_size = batch_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            batches.append(all_particles[start_idx:end_idx])
            start_idx = end_idx
            
        return batches
    
    def run(self):
        """Run the parallel PSO algorithm"""
        pbar = tqdm(range(self.generations), desc='Parallel Particle Swarm Optimisation')
        
        # Create a multiprocessing pool
        pool = mp.Pool(processes=self.num_processes)
        
        for generation in pbar:
            start_time = time.time()  # Start time for the generation
            
            # Prepare batches for parallel evaluation
            batches = self._prepare_batches()
            
            # Evaluate batches in parallel
            eval_func = partial(self._evaluate_particle_batch)
            all_results = pool.map(eval_func, batches)
            
            # Flatten results
            all_particle_fitnesses = []
            subswarm_fitnesses = [[] for _ in range(self.num_sub_swarms)]
            
            # Process results and update particles
            for batch_results in all_results:
                for swarm_idx, particle_idx, fitness in batch_results:
                    all_particle_fitnesses.append(fitness)
                    subswarm_fitnesses[swarm_idx].append(fitness)
                    
                    # Get the particle
                    particle = self.swarms[swarm_idx][particle_idx]
                    
                    # Update particle's personal best
                    if fitness < particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()
                    
                    # Update subswarm's best
                    if fitness < self.subswarm_best_fitnesses[swarm_idx]:
                        self.subswarm_best_fitnesses[swarm_idx] = fitness
                        self.subswarm_best_positions[swarm_idx] = particle['position'].copy()
            
            # Update subswarm best fitness and position arrays
            for swarm_idx in range(self.num_sub_swarms):
                self.subswarm_best_fitness_array[swarm_idx].append(self.subswarm_best_fitnesses[swarm_idx])
                self.subswarm_best_position_array[swarm_idx].append(self.subswarm_best_positions[swarm_idx])
                self.subswarm_avg_array[swarm_idx].append(np.mean(subswarm_fitnesses[swarm_idx]))
                
                # Log histogram of best fitnesses for this subswarm
                self.writer.add_histogram(f'Subswarm_{swarm_idx+1}/Fitnesses', 
                                          np.array(subswarm_fitnesses[swarm_idx]), 
                                          generation)
            
            # Log histogram of best fitnesses for all subswarms
            self.writer.add_histogram('All_Subswarms/Fitnesses', 
                                      np.array(all_particle_fitnesses), 
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
            
            # Log and save periodically
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
            pbar.set_description(f"Parallel PSO - Best: {self.global_best_fitness:.6e} - {self.num_processes} CPUs")
        
        # Clean up
        pool.close()
        pool.join()
        self.writer.flush()
        
        return self.global_best_position, self.global_best_fitness 