import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.mating import Mating

from functions.params import mu, g0, R_earth, w_earth

class ExoAtmosphericPropelledOptimisation:
    def __init__(self,
                 Isp,
                 semi_major_axis,
                 initial_state,
                 structural_mass,
                 mass_flow_exo,
                 mass_payload,
                 burn_time_exo_stage,
                 max_altitude = 100000, # Maximum altitude for the orbit [km]
                 minimum_delta_v_adjustments = 200,
                 number_of_iterations = 200,
                 print_bool = True):
        self.print_bool = print_bool

        self.radius_error = 10  # Error in semi-major axis [m]
        self.minimum_altitude = 80  # Minimum altitude [km]

        self.mu = mu
        self.g0 = g0
        self.Isp = Isp
        self.semi_major_axis = semi_major_axis
        self.initial_state = initial_state # [r_x, r_y, r_z, v_x, v_y, v_z, m]
        self.dry_mass = structural_mass + mass_payload
        self.mass_flow_exo = mass_flow_exo
        self.R_earth = R_earth
        self.thrust = self.Isp * self.g0 * self.mass_flow_exo

        # Scales
        self.time_scale = 1e2   # Time scale for optimization [s]

        self.max_altitude = max_altitude # Maximum altitude for the orbit [km]
        self.minimum_delta_v_adjustments = minimum_delta_v_adjustments  # Minimum delta v adjustments [m/s]

        self.create_bounds(burn_time_exo_stage)
        
        self.initial_optimisation_state = self.generate_initial_optimisation_state(burn_time_exo_stage)

        self.cost_values = []
        self.number_of_iterations = number_of_iterations
        # (number_of_iterations, 7)
        self.optimisation_states = []

    def reset_cost_values(self):
        self.cost_values = []
        self.optimisation_states = []

    def generate_initial_optimisation_state(self, burn_time_exo_stage):
        # self.optimisation_state_bounds
        # Set from upper bounds
        # [prop_time_scaled, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
        initial_optimisation_state = np.zeros(7)

        # Set equal to largest abs bound
        initial_optimisation_state[0] = self.optimisation_state_bounds[0][1]  # prop_time_scaled
        initial_optimisation_state[1] = self.optimisation_state_bounds[1][1]  # pr_x
        initial_optimisation_state[2] = self.optimisation_state_bounds[2][0]  # pr_y
        initial_optimisation_state[3] = self.optimisation_state_bounds[3][1]  # pr_z
        initial_optimisation_state[4] = self.optimisation_state_bounds[4][1]  # pv_x
        initial_optimisation_state[5] = self.optimisation_state_bounds[5][0]  # pv_y
        initial_optimisation_state[6] = self.optimisation_state_bounds[6][1]  # pv_z

        '''
        # Extract initial states
        initial_position_vector = self.initial_state[0:3]
        initial_velocity_vector = self.initial_state[3:6]

        # v_r = v - w_earth x r
        initial_relative_velocity = initial_velocity_vector - np.cross(w_earth, initial_position_vector)
        # pv_hat,0 = v_r/||v_r||
        initial_pvU = initial_relative_velocity / np.linalg.norm(initial_relative_velocity)
        # pr_hat,0 = - omega_orbit x pv_hat,0
        # omega_obit = h_orbit/rp^2 = (r x v)/||r||
        initial_specific_angular_momentum_vector = np.cross(initial_position_vector, initial_velocity_vector)
        initial_orbital_angular_velocity_vector = initial_specific_angular_momentum_vector/(np.linalg.norm(initial_position_vector)**2)
        initial_prU = -np.cross(initial_orbital_angular_velocity_vector, initial_pvU)

        propellant_burn_time_guess = burn_time_exo_stage*3/4 * 1e-2  # Propellant burn time guess [s]

        initial_optimisation_state = [propellant_burn_time_guess,
                                        initial_prU[0],
                                        initial_prU[1],
                                        initial_prU[2],
                                        initial_pvU[0],
                                        initial_pvU[1],
                                        initial_pvU[2]]
        '''
        if self.print_bool:
            print(f'Initial optimisation state: Propellant burn time guess: {initial_optimisation_state[0]*self.time_scale} s \n'
                f'Initial prU: {initial_optimisation_state[1:4]} \n'
                f'Initial pvU: {initial_optimisation_state[4:7]}')
        return initial_optimisation_state

    def create_bounds(self, burn_time_exo_stage):
        self.optimisation_state_bounds = [
            (burn_time_exo_stage/2 * 1/self.time_scale, burn_time_exo_stage * 1/self.time_scale),        # prop_time_scaled
            (0, 0.1),  # pr_x
            (-0.1, 0),  # pr_y
            (0, 0.1),  # pr_z
            (0.9, 1),                           # pv_x
            (-1, 0),                           # pv_y
            (0, 1)                            # pv_z
        ]

    def exo_dyn(self, t, state_vector):
        """
        Defines the ODE system.
        x = [r_x, r_y, r_z, v_x, v_y, v_z, m, p_r_x, p_r_y, p_r_z, p_v_x, p_v_y, p_v_z]
        """
        r = state_vector[0:3]
        v = state_vector[3:6]
        m = state_vector[6]
        p_r = state_vector[7:10]
        p_v = state_vector[10:13]

        r_norm = np.linalg.norm(r)
        p_v_norm = np.linalg.norm(p_v)

        r_dot = v
        v_dot = (-self.mu / r_norm**3) * r + (self.thrust / m) * (p_v / p_v_norm)
        pr_dot = - (self.mu / r_norm**3) * (3 * np.dot(p_v, r) * r / r_norm**2 - p_v)
        pv_dot = -p_r

        m_dot = -self.mass_flow_exo

        dx = np.concatenate((r_dot, v_dot, [m_dot], pr_dot, pv_dot))
        return dx
    
    def final_mass_compute(self,
                           state): # [r_x, r_y, r_z, v_x, v_y, v_z, m]
        """
        Calculate orbital elements from state vector.
        This is a placeholder. Replace with the actual implementation.
        """
        position_vector_final_first_exo_burn = state[0:3]
        velocity_vector_final_first_exo_burn = state[3:6]
        
        mass_final_first_exo_burn = state[6] # Mass after exo's first burn.
        # Compute specific angular momentum
        h_vec = np.cross(position_vector_final_first_exo_burn, velocity_vector_final_first_exo_burn)
        # Compute semimajor axis
        energy = 0.5 * np.linalg.norm(velocity_vector_final_first_exo_burn)**2 - self.mu / np.linalg.norm(position_vector_final_first_exo_burn)
        semi_major_axis = -self.mu / (2 * energy) # Semi-major axis
        # Compute eccentricity
        e_vec = (np.cross(velocity_vector_final_first_exo_burn, h_vec) / self.mu) - (position_vector_final_first_exo_burn / np.linalg.norm(position_vector_final_first_exo_burn))
        eccentricity = np.linalg.norm(e_vec) # Eccentricity

        # Compute final_energy
        final_energy = -self.mu / (2 * semi_major_axis)
        # Compute final_radius at apogee
        final_radius = semi_major_axis * (1 + eccentricity)
        # Compute final velocity for a circular orbit at semi-major axis.
        final_circular_velocity = np.sqrt(self.mu / semi_major_axis)
        # Compute arrival velocity, i.e. after the first burn in exo.
        arrival_velocity = np.sqrt(2 * (final_energy + self.mu / final_radius))
        # Compute change in velocity for a circular impulse
        delta_v_to_circular = final_circular_velocity - arrival_velocity

        mass_final_at_circular = mass_final_first_exo_burn / np.exp(delta_v_to_circular / (self.Isp * self.g0))  # Final mass

        return mass_final_at_circular, final_radius, delta_v_to_circular


    def cost_fcn(self, optimisation_state):
        """
        Objective function for optimization.
        optimisation_state = [prop_time_scaled, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
        """
        propellant_burn_time = optimisation_state[0] * self.time_scale  # Time for burning 2nd stage [s]
        pr = optimisation_state[1:4]
        pv = optimisation_state[4:7]

        augmented_state = np.concatenate((self.initial_state,
                                    pr,
                                    pv))    # Augmented state vector : [r_x, r_y, r_z, v_x, v_y, v_z, m, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
        solution = solve_ivp(self.exo_dyn,
                        t_span = [0, propellant_burn_time],
                        y0 = augmented_state,
                        method='RK45',
                        atol = 1e-8,
                        rtol = 1e-8
                        )

        if not solution.success:
            raise ValueError("ODE integration failed in objective function.")
        
        final_state_vector_arrival = solution.y[0:7, -1]  # Final state vector at arrival : [r_x, r_y, r_z, v_x, v_y, v_z, m]
        mass_final_at_circular, final_radius , _= self.final_mass_compute(final_state_vector_arrival)

        J = -(mass_final_at_circular - self.dry_mass)  # Objective to maximize (negative for minimization)

        J_semimajoraxis = -abs(final_radius - self.semi_major_axis)/1000  # Penalty for semi-major axis error [km]
        J += J_semimajoraxis

        mass_constraint = mass_final_at_circular - self.dry_mass  # >= 0
        altitude_constraint = self.max_altitude - (np.linalg.norm(final_state_vector_arrival[0:3]) - self.R_earth)  # >= 0
        J += altitude_constraint/6
        semi_major_axis_constraint = self.radius_error - abs(final_radius - self.semi_major_axis)
        
        self.cost_values.append(J)
        self.optimisation_states.append(optimisation_state)
        #if self.print_bool:
        #    print(f'Objective: {J}, J_semi_major_axis {J_semimajoraxis:.2f},'
        #        f' mass_constraint: {mass_constraint:.2f}, altitude_constraint: {altitude_constraint:.2f}, semi_major_axis_constraint: {semi_major_axis_constraint:.2f}')
        return J    

    def plot_cost_values(self):
        plt.figure()
        plt.plot(self.cost_values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective function')
        plt.grid()
        plt.show()

    def optimise(self):

        def mass_constraint_fcn(optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.initial_state, pr, pv))
            solution = solve_ivp(self.exo_dyn,
                            t_span = [0, propellant_burn_time],
                            y0 = augmented_state,
                            method='RK45',
                            atol = 1e-8,
                            rtol = 1e-8
                            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_state = solution.y[0:7, -1]
            mass_final_at_circular, _, _ = self.final_mass_compute(final_state)
            mass_constraint = mass_final_at_circular - self.dry_mass # >= 0

            return mass_constraint
        
        def altitude_constraint_fcn(optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.initial_state, pr, pv))
            solution = solve_ivp(self.exo_dyn,
                            t_span = [0, propellant_burn_time],
                            y0 = augmented_state,
                            method='RK45',
                            atol = 1e-8,
                            rtol = 1e-8
                            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")
            
            final_position_exo_arrival = solution.y[0:3, -1]
            final_altitude_exo_arrival = np.linalg.norm(final_position_exo_arrival) - self.R_earth

            # Inequality constraints:  [100000 - final_altitude_exo_arrival,
            #                           self.dry_mass - mass_final_at_circular,
            #                           delta_v_to_circular - 200]
            altitude_constraint = self.max_altitude - final_altitude_exo_arrival  # >= 0

            return altitude_constraint
        
        def final_altitude_constraint(optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.initial_state, pr, pv))
            solution = solve_ivp(self.exo_dyn,
                            t_span = [0, propellant_burn_time],
                            y0 = augmented_state,
                            method='RK45',
                            atol = 1e-8,
                            rtol = 1e-8
                            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_position_exo_arrival = solution.y[0:3, -1]
            final_altitude_exo_arrival = np.linalg.norm(final_position_exo_arrival) - self.R_earth

            minimum_altitude_constraint = final_altitude_exo_arrival - self.minimum_altitude # >= 0

            return minimum_altitude_constraint
        

        def semi_major_axis_constraint_fcn(optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.initial_state, pr, pv))
            solution = solve_ivp(self.exo_dyn,
                            t_span = [0, propellant_burn_time],
                            y0 = augmented_state,
                            method='RK45',
                            atol = 1e-8,
                            rtol = 1e-8
                            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_state = solution.y[0:7, -1]
            _, final_radius, _ = self.final_mass_compute(final_state)
            semi_major_axis_constraint = self.radius_error - abs(final_radius - self.semi_major_axis)  # Equality constraint

            return semi_major_axis_constraint
        
       
        try:
            c_1 = mass_constraint_fcn(self.initial_optimisation_state)
            c_2 = altitude_constraint_fcn(self.initial_optimisation_state)
            c_3 = semi_major_axis_constraint_fcn(self.initial_optimisation_state)
            c_4 = final_altitude_constraint(self.initial_optimisation_state)
            c = np.array([c_1, c_2, c_3, c_4])
            if np.any(c > 0):
                print("Initial guess violates constraints.")
                # Adjust the initial guess or bounds accordingly
        except ValueError as e:
            print(f"Initial guess leads to error in constraints: {e}")
            # Adjust the initial guess or bounds accordingly



        cons = [{'type':'ineq', 'fun': mass_constraint_fcn},
                {'type':'ineq', 'fun': altitude_constraint_fcn},
                {'type':'ineq', 'fun': semi_major_axis_constraint_fcn},
                {'type':'ineq', 'fun': final_altitude_constraint}]

        # Perform optimization
        result = minimize(
            fun=self.cost_fcn,
            x0=self.initial_optimisation_state ,
            bounds=self.optimisation_state_bounds,
            constraints=cons,
            method='trust-constr',
            options={
                'disp': True,
                'maxiter': self.number_of_iterations,
                'initial_tr_radius': 1
            }
        )
        optimised_state = result.x
        propellant_burn_time = optimised_state[0] * self.time_scale  # Optimized propellant time in seconds

        if result.success:
            print("Optimization successful!")
        else:
            print("Optimization did not converge:", result.message)

        self.plot_cost_values()

        
        prU = optimised_state[1:4]
        pvU = optimised_state[4:7]
        if self.print_bool:
            print("Optimization successful!")
            print(f"Optimized propellant time: {propellant_burn_time} s")
            print(f"Optimized pr vector: {prU}")
            print(f"Optimized pv vector: {pvU}")
        # Check it runs and meets constraints
        c_ineq_1 = mass_constraint_fcn(optimised_state)
        c_ineq_2 = altitude_constraint_fcn(optimised_state)
        c_ineq_3 = semi_major_axis_constraint_fcn(optimised_state)
        c_ineq_4 = final_altitude_constraint(optimised_state)
        if self.print_bool:
            print(f"Final constraints: {c_ineq_1} >= 0, {c_ineq_2} >= 0, {c_ineq_3} >= 0, {c_ineq_4} >= 0")
        if c_ineq_1 < 0:
            raise ValueError("Mass constraint violated.")
        elif c_ineq_2 < 0:
            raise ValueError("Altitude constraint violated.")
        elif c_ineq_3 < -1:
            raise ValueError("Semi-major axis constraint violated.")
        elif c_ineq_4 < 0:
            raise ValueError("Final altitude constraint violated.")

        optimised_state[0] = propellant_burn_time # Fix scaling!!!

        return optimised_state
    
    class LauncherOptimizationProblemEA(Problem):
        def __init__(self, launcher):
            self.launcher = launcher
            super().__init__(
                n_var=7,
                n_obj=1,
                n_constr=4,
                xl=np.array([bound[0] for bound in launcher.optimisation_state_bounds]),
                xu=np.array([bound[1] for bound in launcher.optimisation_state_bounds]),
                elementwise_evaluation=True
            )

        def _evaluate(self, X, out, *args, **kwargs):
            pop_size = X.shape[0]

            F = np.zeros(pop_size)
            G = np.zeros((pop_size, 4))  # we have 4 constraints in your code

            for i in range(pop_size):
                x = X[i]   # shape (7,)

                # Extract each portion properly
                propellant_burn_time = x[0]  # single value
                pr = x[1:4]                  # 3 values
                pv = x[4:7]                  # 3 values

                # Evaluate your cost:
                F[i] = self.launcher.cost_fcn(x)

                # Evaluate constraints:
                g1 = self.mass_constraint_fcn(x)
                g2 = self.altitude_constraint_fcn(x)
                g3 = self.semi_major_axis_constraint_fcn(x)
                g4 = self.final_altitude_constraint(x)

                G[i, :] = [-g1, -g2, -g3, -g4]  # example sign convention

            out["F"] = F.reshape(-1, 1) # shape (pop_size, 1)
            out["G"] = G                # shape (pop_size, 4)

        def mass_constraint_fcn(self, optimisation_state):
            # optimisation_state has shape (7,) : [prop_time_scaled, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
            # initial state has shape (7,) : [r_x, r_y, r_z, v_x, v_y, v_z, m]
            # augmented state has shape (13,) : [r_x, r_y, r_z, v_x, v_y, v_z, m, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
            propellant_burn_time = optimisation_state[0] * self.launcher.time_scale
            pr = optimisation_state[1:4]  
            pv = optimisation_state[4:7]

            # Flatten initial state if needed
            initial_state_1d = self.launcher.initial_state.ravel()  # shape (7,)

            # Concatenate
            augmented_state = np.concatenate((initial_state_1d, pr, pv))  # shape (13,)
            solution = solve_ivp(
                self.launcher.exo_dyn,
                t_span=[0, propellant_burn_time],
                y0=augmented_state,
                method='RK45',
                atol=1e-8,
                rtol=1e-8
            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_state = solution.y[0:7, -1]
            mass_final_at_circular, _, _ = self.launcher.final_mass_compute(final_state)
            return mass_final_at_circular - self.launcher.dry_mass  # Constraint should be >= 0

        def altitude_constraint_fcn(self, optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.launcher.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.launcher.initial_state, pr, pv))
            solution = solve_ivp(
                self.launcher.exo_dyn,
                t_span=[0, propellant_burn_time],
                y0=augmented_state,
                method='RK45',
                atol=1e-8,
                rtol=1e-8
            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_position = solution.y[0:3, -1]
            final_altitude = np.linalg.norm(final_position) - self.launcher.R_earth
            return self.launcher.max_altitude - final_altitude  # Constraint should be >= 0

        def final_altitude_constraint(self, optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.launcher.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.launcher.initial_state, pr, pv))
            solution = solve_ivp(
                self.launcher.exo_dyn,
                t_span=[0, propellant_burn_time],
                y0=augmented_state,
                method='RK45',
                atol=1e-8,
                rtol=1e-8
            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_position = solution.y[0:3, -1]
            final_altitude = np.linalg.norm(final_position) - self.launcher.R_earth
            return final_altitude - self.launcher.minimum_altitude  # Constraint should be >= 0
        
        def semi_major_axis_constraint_fcn(self, optimisation_state):
            propellant_burn_time = optimisation_state[0] * self.launcher.time_scale
            pr = optimisation_state[1:4]
            pv = optimisation_state[4:7]

            augmented_state = np.concatenate((self.launcher.initial_state, pr, pv))
            solution = solve_ivp(self.launcher.exo_dyn,
                            t_span = [0, propellant_burn_time],
                            y0 = augmented_state,
                            method='RK45',
                            atol = 1e-8,
                            rtol = 1e-8
                            )

            if not solution.success:
                raise ValueError("ODE integration failed in constraints.")

            final_state = solution.y[0:7, -1]
            _, final_radius, _ = self.launcher.final_mass_compute(final_state)
            semi_major_axis_constraint = self.launcher.radius_error - abs(final_radius - self.launcher.semi_major_axis)  # Equality constraint

            return semi_major_axis_constraint

    def optimise_ea(self):
        def binary_tournament(pop, P, *args, **kwargs):
            # For binary tournament, P is typically shape (n_matings, 2).
            # So we can iterate over each row of P:
            S = np.full(P.shape[0], -1, dtype=int)
            for i in range(P.shape[0]):
                a, b = P[i, 0], P[i, 1]
                if pop[a].F[0] < pop[b].F[0]:
                    S[i] = a
                else:
                    S[i] = b
            return S
        problem = self.LauncherOptimizationProblemEA(self)
        algorithm = GA(
            pop_size=200,  # Increased population size for better exploration
            sampling=FloatRandomSampling(),  # Random initialization
            crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover
            mutation=PM(prob=0.2, eta=20),  # Polynomial Mutation
            selection=TournamentSelection(func_comp=binary_tournament),
            eliminate_duplicates=True,
            mating=Mating(selection=TournamentSelection(func_comp=binary_tournament), crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.2, eta=20), n_offsprings=50)
        )
        termination = get_termination("n_gen", self.number_of_iterations)

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=True
        )

        best_x = res.X
        best_f = res.F[0]
        best_G = res.G

        # Re-scale propellant burn time
        best_x[0] *= self.time_scale

        if self.print_bool:
            print("\n=== EA Optimization Results ===")
            print(f"Best Objective (Cost): {best_f}")
            print(f"Best Variables: {best_x}")
            print(f"Constraint Violations: {best_G}")

        return best_x, best_f, best_G