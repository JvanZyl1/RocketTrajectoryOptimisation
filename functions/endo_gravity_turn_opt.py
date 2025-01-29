from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.mating import Mating
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from Rocket_Trajectory_Class import rocket_trajectory_optimiser, mission_requirements, physical_constants, design_parameters, mission_profile
import numpy as np
import math

class GravityTurnRocket(rocket_trajectory_optimiser):
    def __init__(self,
                 mission_requirements,
                 physical_constants,
                 design_parameters,
                 mission_profile,
                 q_max=1e5,
                 delta_v_losses_estimate=2000,
                 plot_bool=False,
                 save_file_path='/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
        super().__init__(mission_requirements,
                         physical_constants,
                         design_parameters,
                         mission_profile,
                         delta_v_losses_estimate,
                         plot_bool,
                         save_file_path,
                         process_trajectory_bool = False,
                         debug_bool=False)
        self.q_max = q_max  # Maximum dynamic pressure constraint

        self.kick_angle_bounds = (math.radians(0.001),
                                 math.radians(0.5))
        
        self.vertical_rising()

        # Clear state and trajectory results
        self.initial_state = self.state
        self.initial_time = self.time
        self.states = []
        self.times = []

    def reset(self):
        self.state = self.initial_state
        self.time = self.initial_time
        self.states = []
        self.times = []

    
class GravityTurnOptimizationProblem(Problem):
    def __init__(self, rocket):
        super().__init__(n_var=1,
                         n_obj=1,
                         n_constr=1,
                         xl=np.array([rocket.kick_angle_bounds[0]]),
                         xu=np.array([rocket.kick_angle_bounds[1]]))
        self.rocket = rocket

    def find_max_q(self, states):
        # Loop through states and calculate q
        q_values = []
        for state in states:
            position_vector = state[:3]
            velocity_vector = state[3:6]
            altitude = np.linalg.norm(position_vector) - self.rocket.R_earth
            rho, P_a, a = self.rocket.endo_atmospheric_model(altitude)
            q = 0.5 * rho * np.linalg.norm(velocity_vector)**2
            q_values.append(q)

        return max(q_values)

    def _evaluate(self, x, out, *args, **kwargs):
        results = []
        constraints = []
        
        for kick_angle in x:
            self.rocket.kick_angle = kick_angle[0]
            self.rocket.gravity_turn()
            
            final_altitude = np.linalg.norm(self.rocket.state[:3]) - self.rocket.R_earth
            maximum_dynamic_pressure = self.find_max_q(self.rocket.states)
            
            results.append(-final_altitude)  # Negative for minimization
            
            constraints.append(maximum_dynamic_pressure - self.rocket.max_dynamic_pressure)  # Constraint violation
            # Reset
            self.rocket.reset()
        
        out["F"] = np.array(results)
        out["G"] = np.array(constraints)    

class LauncherOptimizationGA:
    def __init__(self, rocket, number_of_iterations=8):
        self.rocket = rocket
        self.number_of_iterations = number_of_iterations

    def optimise_ea(self):
        problem = GravityTurnOptimizationProblem(self.rocket)
        
        def binary_tournament(pop, P, *args, **kwargs):
            S = np.full(P.shape[0], -1, dtype=int)
            for i in range(P.shape[0]):
                a, b = P[i, 0], P[i, 1]
                S[i] = a if pop[a].F[0] < pop[b].F[0] else b
            return S

        algorithm = GA(
            pop_size=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.4, eta=20),
            selection=TournamentSelection(func_comp=binary_tournament),
            eliminate_duplicates=True,
            mating=Mating(selection=TournamentSelection(func_comp=binary_tournament), crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.2, eta=20), n_offsprings=50)
        )

        termination = get_termination("n_gen", self.number_of_iterations)
        res = minimize(problem, algorithm, termination, seed=1, verbose=True)
        
        best_kick_angle = res.X
        best_altitude = -res.F
        best_constraint_violation = res.G
        
        
        
        return best_kick_angle, best_altitude, best_constraint_violation

def optimise_gravity_turn():
    rocket = GravityTurnRocket(mission_requirements, physical_constants, design_parameters, mission_profile)
    launcher = LauncherOptimizationGA(rocket)
    best_kick_angle, best_altitude, best_constraint_violation= launcher.optimise_ea()
    return best_kick_angle, best_altitude, best_constraint_violation