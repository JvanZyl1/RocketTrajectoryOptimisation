from scipy.optimize import minimize
import warnings

'''
Solver types:
1) Nelder-Mead: derivaite-free simplex method which doesn't require gradient calculations.
2) BFGS: Quasi-Newton method using gradients which can other fast convergence on bounded fast problems.
3) L-BFGS-B: Extension on BFGS for higher dimensional problem.
4) COBYLA: Derivative free with non-linear constraints; for when gradient unavaible or unreliable.
5) trust-constr: Constrained optimisation using trust region methods.
'''

class NonHeuristicOptimisers:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.trust_region_bounds_size = 0.1
        max_iter = 50
        disp_bool = True

        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
        self.nelder_mead = lambda x0, bounds : self._suppress_warnings(lambda: minimize(
                                                        fun = self.objective_function,
                                                        x0 = x0,
                                                        method = 'Nelder-Mead',
                                                        bounds = bounds,
                                                        options = {
                                                            'maxiter' : max_iter,
                                                            'maxfev' : max_iter,
                                                            'adaptive' : True,
                                                            'disp' : disp_bool
                                                            }))
        
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs
        self.bfgs = lambda x0, bounds : minimize(fun = self.objective_function,
                                                 x0 = x0,
                                                 method = 'BFGS',
                                                 bounds = bounds,
                                                 options = {
                                                    'max_iter' : max_iter,
                                                    'c1': 1e-4,
                                                    'c2': 0.9,
                                                    'disp' : disp_bool
                                                    })
        
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
        self.L_BFGS_B = lambda x0, bounds : minimize(fun = self.objective_function,
                                                     x0 = x0,
                                                     method = 'L-BFGS-B',
                                                     bounds = bounds,
                                                     options = {
                                                        'max_iter' : max_iter,
                                                        'disp' : disp_bool
                                                        })
        
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html#optimize-minimize-cobyla
        self.cobyla = lambda x0, bounds : minimize(fun = self.objective_function,
                                                   x0 = x0,
                                                   method = 'COBYLA',
                                                   bounds = bounds,
                                                   options = {
                                                        'max_iter' : max_iter,
                                                        'disp' : disp_bool
                                                        })
        
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trust-constr.html#optimize-minimize-trust-constr
        self.trust_constr = lambda x0, bounds : minimize(fun = self.objective_function,
                                                         x0 = x0,
                                                         method = 'trust-constr',
                                                         bounds = bounds,
                                                         options = {
                                                             'max_iter' : max_iter,
                                                             'disp' : disp_bool,
                                                             'initial_tr_radius' : self.trust_region_bounds_size
                                                         })
        
        self.choose_solver('nelder-mead')
        self.iteration = 0
    def choose_solver(self, solver_name):
        if solver_name == 'nelder-mead' or 'Nelder-Mead':
            self.solver = self.nelder_mead
        elif solver_name == 'bfgs' or 'BFGS':
            self.solver = self.bfgs
        elif solver_name == 'l-bfgs-b' or 'L-BFGS-B' or 'L_BFGS_B':
            self.solver = self.L_BFGS_B
        elif solver_name == 'cobyla' or 'COBYLA':
            self.solver = self.cobyla
        elif solver_name == 'trust-constr':
            self.solver = self.trust_constr
        else:
            raise ValueError(f"Solver {solver_name} not found")       

    def objective_function(self, individual):
        fitness = self.model.objective_function(individual)
        self.iteration += 1
        return fitness
    
    def run(self, initial_chromosome):
        print(f'Solving for {initial_chromosome[:5]}...')
        self.iteration = 0
        # Define the bounds for the variables, which is +0.1, - 0.1 the original value
        bounds = [(-0.1 + gene, 0.1 + gene) for gene in initial_chromosome]
        
        # Run the optimisation
        result = self.solver(initial_chromosome, bounds)

        return result.x
        
    def _suppress_warnings(self, func):
        """Helper method to suppress specific warnings during optimization"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Maximum number of function evaluations has been exceeded')
            return func()
        
        