from scipy.optimize import minimize
import warnings
import numpy as np

'''
Solver types:
1) Nelder-Mead: derivaite-free simplex method which doesn't require gradient calculations.
2) BFGS: Quasi-Newton method using gradients which can other fast convergence on bounded fast problems.
3) L-BFGS-B: Extension on BFGS for higher dimensional problem.
4) COBYLA: Derivative free with non-linear constraints; for when gradient unavaible or unreliable.
5) trust-constr: Constrained optimisation using trust region methods.
6) Particle Swarm Optimisation: A population-based optimisation method that uses a swarm of particles to search for the optimal solution.
7) Genetic Algorithm: A population-based optimisation method that uses a population of individuals to search for the optimal solution.
'''
from src.evolutionary_algorithms.particle_swarm_optimisation import ParticleSwarmOptimization

class LocalSearchOptimisers:
    def __init__(self,
                 model,
                 model_name,
                 solver_name,
                 local_search_particle_swarm_optimisation_params = None,
                 max_iter = 1000,
                 trust_region_bounds_size = 0.1):
        self.model = model
        self.model_name = model_name
        self.trust_region_bounds_size = trust_region_bounds_size
        self.disp_bool = True
        self.max_iter = max_iter       
        self.choose_solver(solver_name, local_search_particle_swarm_optimisation_params)
        self.solver_name = solver_name
        self.iteration = 0

    def choose_solver(self, solver_name, local_search_particle_swarm_optimisation_params):
        if solver_name in ['nelder-mead', 'Nelder-Mead']:
            print(f'Using Nelder-Mead for local search')
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
            self.solver = lambda x0, bounds : self._suppress_warnings(lambda: minimize(
                                                        fun = self.objective_function,
                                                        x0 = x0,
                                                        method = 'Nelder-Mead',
                                                        bounds = bounds,
                                                        options = {
                                                            'maxiter': self.max_iter,
                                                            'maxfev': self.max_iter,
                                                            'adaptive': True,
                                                            'disp': self.disp_bool,
                                                            'xatol': 1e-8,          # Absolute tolerance for x
                                                            'fatol': 1e-8           # Absolute tolerance for f
                                                            }))
        elif solver_name in ['bfgs', 'BFGS']:
            print(f'Using BFGS for local search')
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs
            self.solver = lambda x0, bounds : minimize(fun = self.objective_function,
                                                 x0 = x0,
                                                 method = 'BFGS',
                                                 bounds = bounds,
                                                 options = {
                                                    'max_iter' : self.max_iter,
                                                    'c1': 1e-4,
                                                    'c2': 0.9,
                                                    'disp' : self.disp_bool
                                                    })
        elif solver_name in ['l-bfgs-b', 'L-BFGS-B', 'L_BFGS_B']:
            print(f'Using L-BFGS-B for local search')
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
            self.solver = lambda x0, bounds : minimize(fun = self.objective_function,
                                                     x0 = x0,
                                                     method = 'L-BFGS-B',
                                                     bounds = bounds,
                                                     options = {
                                                        'max_iter' : self.max_iter,
                                                        'disp' : self.disp_bool
                                                        })
        elif solver_name in ['cobyla', 'COBYLA']:
            print(f'Using COBYLA for local search')
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html#optimize-minimize-cobyla
            self.solver = lambda x0, bounds : minimize(fun = self.objective_function,
                                                   x0 = x0,
                                                   method = 'COBYLA',
                                                   bounds = bounds,
                                                   options = {
                                                        'max_iter' : self.max_iter,
                                                        'disp' : self.disp_bool
                                                        })

        elif solver_name == 'trust-constr':
            print(f'Using Trust-Constr for local search')
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trust-constr.html#optimize-minimize-trust-constr
            self.solver = lambda x0, bounds : minimize(fun = self.objective_function,
                                                         x0 = x0,
                                                         method = 'trust-constr',
                                                         bounds = bounds,
                                                         options = {
                                                             'max_iter' : self.max_iter,
                                                             'disp' : self.disp_bool,
                                                             'initial_tr_radius' : self.trust_region_bounds_size
                                                         })
        elif solver_name == 'particle_swarm_optimisation':
            assert local_search_particle_swarm_optimisation_params is not None, "Particle Swarm Optimisation parameters must be provided"
            print(f'Using Particle Swarm Optimisation for local search')
            def particle_swarm_run_lambda_func(x0, bounds):
                pso = ParticleSwarmOptimization(pso_params = local_search_particle_swarm_optimisation_params,
                                                bounds = bounds,
                                                model = self.model,
                                                model_name = self.model_name,
                                                local_search_optimiser = None,
                                                local_search_plot_bool = True)
                pso.swarm[0]['position'] = x0
                pso.swarm[0]['velocity'] = np.zeros(len(x0))
                pso.run()
                return pso.global_best_position
            
            self.solver = lambda x0, bounds : particle_swarm_run_lambda_func(x0, bounds)
        else:
            raise ValueError(f"Solver {solver_name} not found")       

    def objective_function(self, individual):
        fitness = self.model.objective_function(individual)
        self.iteration += 1
        print(f'Iteration: {self.iteration}, & Fitness: {fitness}')
        return fitness
    
    def run(self, initial_chromosome):
        print(f'Solving for {initial_chromosome[:5]}...')
        self.iteration = 0
        
        # Define wider bounds to give the optimizer more room to explore
        # Current bounds are too tight at just ±0.1 around each value
        bounds = [(gene - self.trust_region_bounds_size, gene + self.trust_region_bounds_size) for gene in initial_chromosome]
        
        # Run the optimisation
        result = self.solver(initial_chromosome, bounds)

        # New chromosome:
        if self.solver_name == 'particle_swarm_optimisation':
            new_chromosome = result
        else:
            new_chromosome = result.x

        print(f'Resulting in: {new_chromosome[:5]}...')
        print(f'Improvement: {self.model.objective_function(initial_chromosome) - self.model.objective_function(new_chromosome)}')
        # Average gene change
        gene_change = np.mean(np.abs(np.array(new_chromosome) - np.array(initial_chromosome)))
        print(f'Average gene change: {gene_change}')

        return new_chromosome
        
    def _suppress_warnings(self, func):
        """Helper method to suppress specific warnings during optimization"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Maximum number of function evaluations has been exceeded')
            return func()
        

# Configure PSO, to have local search

class ParticleSwarmOptimization_with_local_search():
    def __init__(self,
                 pso_params,
                 global_bounds,
                 model,
                 model_name):
        local_search_optimiser = LocalSearchOptimisers(model,
                                                       model_name,
                                                       solver_name = pso_params['local_search_solver'],
                                                       local_search_particle_swarm_optimisation_params = pso_params['local'],
                                                       max_iter = pso_params['local_search_max_iter'],
                                                       trust_region_bounds_size = pso_params['local_search_trust_region_bounds_size'])
        
        self.pso_global_optimiser = ParticleSwarmOptimization(pso_params = pso_params,
                                                              bounds = global_bounds,
                                                              model = model,
                                                              model_name = model_name,
                                                              local_search_optimiser = local_search_optimiser)
        
    def run(self):
        self.pso_global_optimiser.run()
        return self.pso_global_optimiser.global_best_position
        
    def reset(self):
        self.pso_global_optimiser.reset()

    def plot_convergence(self):
        self.pso_global_optimiser.plot_convergence(self.model_name)

    def plot_results(self):
        self.pso_global_optimiser.plot_results()