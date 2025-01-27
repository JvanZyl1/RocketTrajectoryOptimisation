import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import math

from params import mu, g0, R_earth, w_earth

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
        self.verify_initial_opt_state()

        self.cost_values = []
        self.number_of_iterations = number_of_iterations
        # (number_of_iterations, 7)
        self.optimisation_states = []

    def reset_cost_values(self):
        self.cost_values = []
        self.optimisation_states = []

    def generate_initial_optimisation_state(self, burn_time_exo_stage):
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
        if self.print_bool:
            print(f'Initial optimisation state: Propellant burn time guess: {propellant_burn_time_guess} s \n'
                f'Initial prU: {initial_prU} \n'
                f'Initial pvU: {initial_pvU}')
        return initial_optimisation_state
    
    def verify_initial_opt_state(self):
        try:
            c, ceq = self.constraint_fcn(self.initial_optimisation_state)
            if np.any(c > 0) or not np.isclose(ceq, 0, atol=1e-5):
                print("Initial guess violates constraints.")
                # Adjust the initial guess or bounds accordingly
        except ValueError as e:
            print(f"Initial guess leads to error in constraints: {e}")
            # Adjust the initial guess or bounds accordingly



    def create_bounds(self, burn_time_exo_stage):
        self.optimisation_state_bounds = [
            (burn_time_exo_stage/2 * 1/self.time_scale, burn_time_exo_stage * 1/self.time_scale),        # prop_time_scaled
            (-1, 1),  # pr_x
            (-1, 1),  # pr_y
            (-1, 1),  # pr_z
            (-1, 1),                           # pv_x
            (-1, 1),                           # pv_y
            (-1, 1)                            # pv_z
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
        if np.linalg.norm(p_v) == 0:
            p_v_norm = 1e-8
        else:
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
        mass_final_at_circular, _ , _= self.final_mass_compute(final_state_vector_arrival)

        J = -(mass_final_at_circular - self.dry_mass)*100  # Objective to maximize (negative for minimization)

        self.cost_values.append(J)
        self.optimisation_states.append(optimisation_state)
        if self.print_bool:
            print(f'Objective: {J}, constraint ineq 1: {mass_final_at_circular - self.dry_mass} >= 0, mass_at_arrival: {final_state_vector_arrival[6]}')
        return J

    def mass_constraint(self, optimisation_state):
        """
        Constraint function for optimization.
        Returns a tuple (c, ceq) where c <= 0 and ceq == 0.
        """
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

    def constraint_fcn(self, optimisation_state):
        """
        Constraint function for optimization.
        Returns a tuple (c, ceq) where c <= 0 and ceq == 0.
        """
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
        final_position_exo_arrival = solution.y[0:3, -1]
        final_altitude_exo_arrival = np.linalg.norm(final_position_exo_arrival) - self.R_earth

        mass_final_at_circular, final_radius, delta_v_to_circular = self.final_mass_compute(final_state)
        c_equality = (final_radius - self.semi_major_axis)  # Equality constraint

        # Inequality constraints:  [100000 - final_altitude_exo_arrival,
        #                           self.dry_mass - mass_final_at_circular,
        #                           delta_v_to_circular - 200]
        c_ineq_1 = self.max_altitude - final_altitude_exo_arrival  # >= 0
        c_ineq_2 = mass_final_at_circular - self.dry_mass # >= 0
        #c_ineq_3 = delta_v_to_circular - self.minimum_delta_v_adjustments # >= 0

        return c_ineq_1, c_ineq_2, c_equality
    def optimise(self):
        """
        Perform the optimization using scipy.optimize.minimize.
        """
        # Define constraints in scipy's format
        ineq_constraints = {
            'type': 'ineq',
            'fun': lambda x: self.constraint_fcn(x)[:]  # Directly use c >= 0
        }

        # Combine constraints
        all_constraints = [ineq_constraints]

        # Perform optimization
        result = minimize(
            fun=self.cost_fcn,
            x0=self.initial_optimisation_state ,
            bounds=self.optimisation_state_bounds,
            constraints=all_constraints,
            method='SLSQP',
            options={
                'disp': True,
                'maxiter': self.number_of_iterations
            }
        )
        optimised_state = result.x
        propellant_burn_time = optimised_state[0] * self.time_scale  # Optimized propellant time in seconds
        
        prU = optimised_state[1:4]
        pvU = optimised_state[4:7]
        print("Optimization successful!")
        print(f"Optimized propellant time: {propellant_burn_time} s")
        print(f"Optimized pr vector: {prU}")
        print(f"Optimized pv vector: {pvU}")
        # Check it runs and meets constraints
        c_ineq_1, c_ineq_2, ceq = self.constraint_fcn(optimised_state)
        print(f"Final constraints: {c_ineq_1} >= 0, {c_ineq_2} >= 0, {ceq} = 0")
        if c_ineq_1 < 0 or c_ineq_2 < 0 or ceq != 0:
            raise ValueError("Constraints not met.")

        optimised_state[0] = propellant_burn_time # Fix scaling!!!

        return optimised_state