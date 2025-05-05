import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.gridspec as gridspec
from torch.utils.tensorboard import SummaryWriter
from src.envs.utils.atmosphere_dynamics import gravity_model_endo, endo_atmospheric_model

def objective(x, callback, m0=None):
    # If the input is a 2D array (multiple particles)
    if x.ndim > 1:
        j = []
        for i in range(x.shape[0]):
            # Extract throttle profile and burn time for each particle
            u = x[i, :-1]
            T_burn = x[i, -1]
            
            # Run simulation
            sim_results = callback.run_simulation(tuple(u.tolist() + [T_burn]))
            
            # Enhanced smoothness penalties
            # 1. First derivative penalty (throttle rate changes)
            first_derivative_penalty = np.sum(np.diff(u)**2) / T_burn
            
            # 2. Second derivative penalty (throttle acceleration changes)
            second_derivative = np.diff(np.diff(u))
            second_derivative_penalty = np.sum(second_derivative**2) / T_burn if len(second_derivative) > 0 else 0
            
            # Combine smoothness penalties
            smoothness_penalty = first_derivative_penalty * 10 + second_derivative_penalty * 15
            
            # Calculate the objective value
            # For 8 engines, we need to prioritize minimizing propellant usage more heavily
            objective_value = (m0 - sim_results[2]) / 10000 + smoothness_penalty*1e6
            
            # Calculate constraint violations
            y_final = sim_results[0][-1]
            v_final = sim_results[1][-1]
            min_mp = np.min(sim_results[4])
            vy_dot = sim_results[5]
            
            # Add penalties for constraint violations
            y_violation = max(0, callback.yf_min - y_final, y_final - callback.yf_max)
            v_violation = max(0, callback.vf_min - abs(v_final), abs(v_final) - callback.vf_max)
            
            dynamic_violations = np.minimum(sim_results[3], 0)
            max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
            
            prop_violation = max(0, -min_mp)
            vy_dot_violation = max(0, abs(vy_dot) - callback.vy_dot_margin)
            
            # Add heavy penalty for constraint violations
            constraint_penalty = 1e6 * (y_violation + v_violation + max_dynamic_violation + prop_violation + vy_dot_violation)
            
            j.append(objective_value + constraint_penalty)
        
        return np.array(j)
    else:
        # Handle single particle case
        u = x[:-1]
        T_burn = x[-1]
        
        # Run simulation
        sim_results = callback.run_simulation(tuple(u.tolist() + [T_burn]))
        
        # Enhanced smoothness penalties
        # 1. First derivative penalty (throttle rate changes)
        first_derivative_penalty = np.sum(np.diff(u)**2) / T_burn
        
        # 2. Second derivative penalty (throttle acceleration changes)
        second_derivative = np.diff(np.diff(u))
        second_derivative_penalty = np.sum(second_derivative**2) / T_burn if len(second_derivative) > 0 else 0
        
        # Combine smoothness penalties
        smoothness_penalty = first_derivative_penalty * 10 + second_derivative_penalty * 15
        
        # Calculate the objective value - adjusted for 8 engines
        objective_value = (m0 - sim_results[2]) / 10000 + smoothness_penalty
        
        # Add penalties for constraint violations
        y_final = sim_results[0][-1]
        v_final = sim_results[1][-1]
        min_mp = np.min(sim_results[4])
        vy_dot = sim_results[5]
        
        # Add penalties for constraint violations
        y_violation = max(0, callback.yf_min - y_final, y_final - callback.yf_max)
        v_violation = max(0, callback.vf_min - abs(v_final), abs(v_final) - callback.vf_max)
        
        dynamic_violations = np.minimum(sim_results[3], 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        
        prop_violation = max(0, -min_mp)
        vy_dot_violation = max(0, abs(vy_dot) - callback.vy_dot_margin)
        
        # Add heavy penalty for constraint violations
        constraint_penalty = 1e6 * (y_violation + v_violation + max_dynamic_violation + prop_violation + vy_dot_violation)
        
        return objective_value + constraint_penalty

# Callback class to track optimization progress
class OptimisationCallback:
    def __init__(self,
                 simulate_function,
                 m0,
                 yf_min,
                 yf_max,
                 vf_min,
                 vf_max,
                 vy_dot_margin):
        self.iterations = []
        self.objectives = []
        self.alt_violations = []
        self.vel_violations = []
        self.dyn_violations = []
        self.prop_violations = []
        self.burn_times = []
        self.final_alts = []
        self.final_vels = []
        self.min_props = []
        self.vy_dot_violations = []  # Add list to store vy_dot violations
        self.simulation_cache = {}
        self.simulate_function = simulate_function
        self.m0 = m0
        self.yf_min = yf_min
        self.yf_max = yf_max
        self.vf_min = vf_min
        self.vf_max = vf_max
        self.vy_dot_margin = vy_dot_margin

        # Set up TensorBoard logging
        log_dir = 'data/reference_trajectory/landing_burn_controls/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        
        # Current iteration
        self.iter_count = 0

    def run_simulation(self, x):
        if isinstance(x, tuple):
            # If x is already a tuple, use it directly
            key = x
        else:
            # Otherwise, convert it to a tuple for hashing
            key = tuple(x)
            
        if key not in self.simulation_cache:
            if isinstance(x, tuple):
                u = np.array(x[:-1])
                T_burn = x[-1]
            else:
                u = x[:-1]
                T_burn = x[-1]
                
            # Apply smoothing to the throttle profile before simulation
            # This helps to ensure that even if the optimizer tries to apply rapid changes,
            # we get a physically realizable throttle profile
            
            # Method 1: Apply a moving average filter
            window_size = 3  # Small window to preserve important features
            if len(u) >= window_size:
                u_smooth = np.zeros_like(u)
                # Keep the edges unchanged
                u_smooth[:window_size//2] = u[:window_size//2]
                u_smooth[-window_size//2:] = u[-window_size//2:]
                
                # Apply moving average to the central part
                for i in range(window_size//2, len(u) - window_size//2):
                    u_smooth[i] = np.mean(u[i-window_size//2:i+window_size//2+1])
                
                u = u_smooth
            
            self.simulation_cache[key] = self.simulate_function(u, T_burn)
        return self.simulation_cache[key]

    def update_logs(self, best_pos):
        x = best_pos
        sim_results = self.run_simulation(tuple(x.tolist()))
        y_final = sim_results[0][-1]
        v_final = sim_results[1][-1]
        min_mp = np.min(sim_results[4])
        vy_dot = sim_results[5]
        
        y_violation = max(0, self.yf_min - y_final, y_final - self.yf_max)
        v_violation = max(0, self.vf_min - abs(v_final), abs(v_final) - self.vf_max)
        
        dynamic_violations = np.minimum(sim_results[3], 0)
        max_dynamic_violation = abs(np.min(dynamic_violations)) if np.any(dynamic_violations < 0) else 0
        
        prop_violation = max(0, -min_mp)
        vy_dot_violation = max(0, abs(vy_dot) - self.vy_dot_margin)
        
        # Calculate objective value without penalties using the new smoothness calculation
        u = x[:-1]
        T_burn = x[-1]
        
        # First derivative penalty (throttle rate changes)
        first_derivative_penalty = np.sum(np.diff(u)**2) / T_burn
        
        # Second derivative penalty (throttle acceleration changes)
        second_derivative = np.diff(np.diff(u))
        second_derivative_penalty = np.sum(second_derivative**2) / T_burn if len(second_derivative) > 0 else 0
        
        # Combine smoothness penalties
        smoothness_penalty = first_derivative_penalty * 10 + second_derivative_penalty * 15
        
        obj_val = (self.m0 - sim_results[2]) / 10000 + smoothness_penalty
        
        # Store data
        self.iterations.append(self.iter_count)
        self.objectives.append(obj_val)
        self.alt_violations.append(y_violation)
        self.vel_violations.append(v_violation)
        self.dyn_violations.append(max_dynamic_violation)
        self.prop_violations.append(prop_violation)
        self.burn_times.append(x[-1])
        self.final_alts.append(y_final)
        self.final_vels.append(v_final)
        self.min_props.append(min_mp)
        self.vy_dot_violations.append(vy_dot_violation)
        
        # Log to TensorBoard
        self.writer.add_scalar('Objective', obj_val, self.iter_count)
        self.writer.add_scalar('Final Altitude Violation', y_violation, self.iter_count)
        self.writer.add_scalar('Final Velocity Violation', v_violation, self.iter_count)
        self.writer.add_scalar('Max Dynamic Pressure Violation', max_dynamic_violation, self.iter_count)
        self.writer.add_scalar('Propellant Mass Violation', prop_violation, self.iter_count)
        self.writer.add_scalar('Burn Time', x[-1], self.iter_count)
        self.writer.add_scalar('Final Altitude', y_final, self.iter_count)
        self.writer.add_scalar('Final Velocity', v_final, self.iter_count)
        self.writer.add_scalar('Minimum Propellant Mass', min_mp, self.iter_count)
        self.writer.add_scalar('Vy Dot Violation', vy_dot_violation, self.iter_count)
        self.writer.add_scalar('Smoothness Penalty', smoothness_penalty, self.iter_count)
        
        self.iter_count += 1
        
        return obj_val

def simulate(u, T_end, N, y0, vy0, m0, mp0, Q_max, S_grid_fins, C_n_0, T_e, n_e, v_ex, minimum_throttle, nozzle_exit_pressure, nozzle_exit_area):
    dt = T_end / N
    y = y0
    vy = vy0
    m = m0
    mp = mp0
    t = 0
    y_vals, vy_vals, dynamic_pressure_margins, mp_vals = [], [], [], []
    
    # With 8 engines, we need a very fine-grained throttle control
    # Let's ensure the throttle profile is physically realizable by applying exponential smoothing
    alpha = 0.7  # Smoothing factor
    u_smoothed = np.zeros_like(u)
    u_smoothed[0] = u[0]
    for i in range(1, len(u)):
        u_smoothed[i] = alpha * u[i] + (1 - alpha) * u_smoothed[i-1]
    
    u = u_smoothed  # Use the smoothed throttle profile
    
    for ui in u:
        ui = np.clip(ui, 0, 1) # just in case
        throttle = ui * (1 - minimum_throttle) + minimum_throttle
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        dynamic_pressure_margin = Q_max - 0.5 * density * vy**2
        dynamic_pressure_margins.append(dynamic_pressure_margin)
        force_aero = 0.5 * S_grid_fins * C_n_0 * density * vy**2
        mdot = -throttle * T_e * n_e / v_ex
        g = gravity_model_endo(y)
        T_e_losses = T_e + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area
        vy_dot = -g + throttle * T_e_losses * n_e / m + force_aero / m
        y += vy * dt
        vy += vy_dot * dt
        m += mdot * dt
        mp += mdot * dt
        t += dt
        y_vals.append(y)
        vy_vals.append(vy)
        mp_vals.append(mp)
    return np.array(y_vals), np.array(vy_vals), m, np.array(dynamic_pressure_margins), np.array(mp_vals), vy_dot

class LandingBurnOptimiser:
    def __init__(self):
        self.load_params()
        self.load_initial_conditions()
        # Discretisation
        self.N = 20             # Increased discretization for finer control
        self.max_iter = 50
        self.number_of_particles = 800
        # Bounds - higher burn time for softer landing with more engines
        self.T_burn_min = 140.0    # minimum burn time [s]
        self.T_burn_max = 220.0    # maximum burn time [s]

        # Landing conditions
        self.yf_min = 4.75
        self.yf_max = 5.0
        self.vf_min = -0.05
        self.vf_max = 0.05
        self.vy_dot_margin = 0.01

        self.minimum_throttle = 0.1 # BEUN

        # Initial guess - adjusted for 8 engines
        # With 8 engines and 0.1 minimum throttle, we need much lower throttle values
        u0 = np.zeros(self.N)
        
        # Create a sigmoid-based initial throttle profile that starts very low and gradually increases
        t = np.linspace(0, 1, self.N)
        u0 = 0.2 / (1 + np.exp(-10 * (t - 0.7)))  # Sigmoid that stays low for most of the profile
        
        T_burn0 = 160.0
        self.x0 = np.append(u0, [T_burn0])

        # Bounds for PSO
        self.bounds = (
            np.array([0.0] * self.N + [self.T_burn_min]),  # Lower bounds
            np.array([1.0] * self.N + [self.T_burn_max])   # Upper bounds
        )
        
        # Pass the simulation function directly to the callback
        self.simulation_func_lambda = lambda u, T_end: simulate(u,
                                                                T_end,
                                                                N = self.N,
                                                                y0 = self.y0,
                                                                vy0 = self.vy0,
                                                                m0 = self.m0,
                                                                mp0 = self.mp0,
                                                                Q_max = self.Q_max,
                                                                S_grid_fins = self.S_grid_fins,
                                                                C_n_0 = self.C_n_0,
                                                                T_e = self.T_e,
                                                                n_e = self.n_e,
                                                                v_ex = self.v_ex,
                                                                minimum_throttle = self.minimum_throttle,
                                                                nozzle_exit_pressure = self.nozzle_exit_pressure,
                                                                nozzle_exit_area = self.nozzle_exit_area)
        self.callback = OptimisationCallback(self.simulation_func_lambda, self.m0, self.yf_min, self.yf_max, self.vf_min, self.vf_max, self.vy_dot_margin)
    
    def objective_function(self, x, **kwargs):
        return objective(x, self.callback, self.m0)

    def __call__(self):
        # Options for PSO - updated for better convergence with 8 engines
        options = {
            'c1': 2.0,    # Increased cognitive parameter for exploration
            'c2': 2.0,    # Increased social parameter
            'w': 0.6,     # Reduced inertia for faster convergence 
            'k': 15,      # Increased neighbors for better information sharing
            'p': 2        # Minkowski p-norm
        }
        
        # Create custom initial positions with diversity but encouraging smoothness
        init_positions = []
        
        # Start with the initial guess
        init_positions.append(self.x0)
        
        # For 8 engines, focus on very low throttle values
        # In multiple attempts at solution, we'll concentrate on the space of low throttle values
        
        # Create smooth throttle profiles for initialization
        # For 8 engines and 0.1 minimum throttle, we need much lower throttle values
        for i in range(20):
            # Very low throttle profile (0.05-0.15 range)
            t = np.linspace(0, 1, self.N)
            base_throttle = 0.05 + 0.1 * np.random.random()  # Base value between 0.05 and 0.15
            variation_magnitude = 0.03 * np.random.random()  # Small variations
            
            # Create a smoothly varying profile based on sine waves of different frequencies
            freq1 = 1 + 2 * np.random.random()  # Frequency between 1-3
            freq2 = 0.5 + 1.5 * np.random.random()  # Frequency between 0.5-2
            phase = 2 * np.pi * np.random.random()  # Random phase
            
            smooth_profile = base_throttle + variation_magnitude * (
                np.sin(freq1 * 2 * np.pi * t + phase) + 
                0.5 * np.sin(freq2 * 2 * np.pi * t + 2*phase)
            )
            
            throttle_variation = np.clip(smooth_profile, 0, 0.3)
            burn_time = np.clip(
                self.T_burn_min + (self.T_burn_max - self.T_burn_min) * np.random.random(), 
                self.T_burn_min, 
                self.T_burn_max
            )
            init_positions.append(np.append(throttle_variation, burn_time))
        
        # Add variations around the initial guess for remaining particles
        for _ in range(self.number_of_particles - len(init_positions)):
            # Random variation around initial guess with smoothness
            random_variation = np.random.normal(0, 0.04, self.N)
            # Apply smoothing to the random variation to make it more continuous
            smooth_variation = np.zeros_like(random_variation)
            window = 9  # Increased smoothing window
            for i in range(self.N):
                # Average over a window centered at i
                start = max(0, i - window//2)
                end = min(self.N, i + window//2 + 1)
                smooth_variation[i] = np.mean(random_variation[start:end])
            
            throttle_variation = np.clip(self.x0[:-1] + smooth_variation, 0, 0.3)
            
            # Add some randomness to burn time within bounds
            burn_time_variation = np.clip(
                self.x0[-1] + np.random.normal(0, 15),
                self.T_burn_min,
                self.T_burn_max
            )
            
            # Combine throttle and burn time
            init_positions.append(np.append(throttle_variation, burn_time_variation))
        
        # Convert to numpy array
        init_pos = np.array(init_positions)
        
        # Create the optimizer with the initial positions
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.number_of_particles,
            dimensions=self.N + 1,
            options=options,
            bounds=self.bounds,
            init_pos=init_pos
        )
        
        # Run the optimizer with focused iterations
        print(f"Starting optimization with {self.number_of_particles} particles and max {self.max_iter} iterations...")
        print(f"Initial conditions: altitude={self.y0:.2f}m, velocity={self.vy0:.2f}m/s, mass={self.m0:.2f}kg")
        print(f"Target conditions: altitude={self.yf_min:.2f}-{self.yf_max:.2f}m, velocity={self.vf_min:.2f}-{self.vf_max:.2f}m/s")
        
        # Optimization process with custom logging and early stopping
        def custom_logger(swarm):
            best_pos = swarm.position[swarm.pbest_cost.argmin()]
            cost = self.callback.update_logs(best_pos)
            
            # Set a maximum number of iterations to prevent excessive runtime
            if self.callback.iter_count >= self.max_iter:
                print("Maximum iteration limit reached, stopping optimization.")
                return True
            
            # Check for early convergence - reduced sensitivity for 8 engines
            if self.callback.iter_count > 30:
                last_15_costs = self.callback.objectives[-15:]
                variation = np.std(last_15_costs) / np.mean(last_15_costs) if np.mean(last_15_costs) != 0 else 0
                last_15_alt_violations = self.callback.alt_violations[-15:]
                last_15_vel_violations = self.callback.vel_violations[-15:]
                
                # Only stop early if we have very little variation in cost AND constraint violations are small
                if (variation < 1e-4 and 
                    np.mean(last_15_alt_violations) < 1e-2 and 
                    np.mean(last_15_vel_violations) < 1e-2):
                    print("Early convergence detected, stopping optimization")
                    return True
            
            # Print current iteration progress every 5 iterations
            if self.callback.iter_count % 5 == 0:
                print(f"Iteration {self.callback.iter_count}: Best cost = {cost:.4e}, Best position = [..., T_burn = {best_pos[-1]:.2f}]")
                print(f"  Final altitude = {self.callback.final_alts[-1]:.2f}, Final velocity = {self.callback.final_vels[-1]:.4f}")
            
            return False  # Continue optimization

        cost, pos = optimizer.optimize(
            self.objective_function, 
            iters=self.max_iter,  # Reduced iterations with maximum iteration check in the callback
            verbose=True,
            callback=custom_logger
        )
        
        # Post-process the best solution for additional smoothing
        u_best = pos[:-1]
        T_burn_best = pos[-1]
        
        # Get the best solution
        result = {
            'x': pos,
            'fun': cost
        }
        
        self.save_all(result, optimizer)

    def save_all(self, result, optimizer):
        self.save_optimisation_data()
        self.plot_optimisation_history()
        self.plot_cost_history(optimizer)

        # Output results
        x_opt = result['x']
        u_opt = x_opt[:-1]
        T_end_opt = x_opt[-1]
        self.save_final_parameters(u_opt, T_end_opt)
        self.plot_simulation_results(u_opt, T_end_opt)

    def load_params(self):
        # Load parameters from rocket parameters
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        self.T_e = float(sizing_results['Thrust engine stage 1'])
        self.n_e = 8                                  # Select number of landing burn engines
        self.v_ex = float(sizing_results['Exhaust velocity stage 1'])
        self.Q_max = 30000.0     # max dynamic pressure [Pa]
        self.m_strc = float(sizing_results['Actual structural mass stage 1'])*1000
        self.S_grid_fins = float(sizing_results['S_grid_fins'])
        self.C_n_0 = float(sizing_results['C_n_0'])
        self.nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1'])
        self.nozzle_exit_area = float(sizing_results['Nozzle exit area'])
        # ass. alpha effective = 0 and delta = 0

    def load_initial_conditions(self):
        # Load initial conditions
        file_path_re_entry_data = 'data/reference_trajectory/re_entry_burn_controls/state_action_re_entry_burn_control.csv'
        re_entry_data = pd.read_csv(file_path_re_entry_data)
        self.y0 = re_entry_data['y[m]'].iloc[-1]    # initial altitude [m]
        self.vy0 = re_entry_data['vy[m/s]'].iloc[-1]  # initial vertical velocity [m/s]
        self.m0 = re_entry_data['mass[kg]'].iloc[-1]   # initial mass [kg]
        self.mp0 = self.m0 - self.m_strc          # propellant mass [kg]
    
    def plot_cost_history(self, optimizer):
        plt.figure(figsize=(10, 6))
        plot_cost_history(optimizer.cost_history)
        plt.title("PSO Cost History", fontsize=20)
        plt.xlabel("Iterations", fontsize=18)
        plt.ylabel("Cost", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig('results/classical_controllers/landing_burn_pso_cost_history.png')
        plt.close()
    
    def plot_optimisation_history(self):
        plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.callback.iterations, self.callback.objectives, color='blue', label='Objective', linewidth=4)
        ax1.set_xlabel('Iteration', fontsize=20)
        ax1.set_ylabel('Objective Value', fontsize=20)
        ax1.set_title('Objective Value', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid(True)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2.semilogy(self.callback.iterations, self.callback.alt_violations, color='green', label='Altitude', linewidth=4)
        ax2.semilogy(self.callback.iterations, self.callback.vel_violations, color='red', label='Velocity', linewidth=4)
        ax2.semilogy(self.callback.iterations, self.callback.dyn_violations, color='blue', label='Dynamic Pressure', linewidth=4)
        ax2.semilogy(self.callback.iterations, self.callback.prop_violations, color='black', label='Propellant', linewidth=4)
        ax2.semilogy(self.callback.iterations, self.callback.vy_dot_violations, color='yellow', label='Vy Dot', linewidth=4)
        ax2.set_xlabel('Iteration', fontsize=20)
        ax2.set_ylabel('Constraint Violation', fontsize=20)
        ax2.set_title('Constraint Violations', fontsize=22)
        ax2.legend(fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid(True)
        plt.savefig('results/classical_controllers/landing_burn_optimisation_history.png')
        plt.close()

    def plot_simulation_results(self, u_opt, T_burn_opt):
        ys, vys, m_final, margins, mps, vy_dot = self.simulation_func_lambda(u_opt, T_burn_opt)
        u_opt = np.clip(u_opt, 0, 1) # just in case
        throttle_opt = u_opt * (1 - self.minimum_throttle) + self.minimum_throttle
        t = np.linspace(0, T_burn_opt, self.N)

        # save data to csv
        total_masses = self.m_strc + mps
        os.makedirs('data/reference_trajectory/landing_burn_controls', exist_ok=True)
        with open('data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_simulation_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time [s]', 'Altitude [m]', 'Velocity [m/s]', 'Non-NominalThrottle', 'Propellant Mass [kg]', 'Total Mass [kg]'])
            for i in range(self.N):
                writer.writerow([t[i], ys[i], vys[i], throttle_opt[i], mps[i], total_masses[i]])
        
        # Create a figure with 2x2 grid
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Plot 1: Altitude vs Time
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(t, ys, color='blue', label='Altitude', linewidth=4)
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel('Altitude [m]', fontsize=20)
        ax1.set_title('Altitude', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid(True)
        
        # Plot 2: Velocity vs Time
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(t, vys, color='red', label='Velocity', linewidth=4)
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_ylabel('Velocity [m/s]', fontsize=20)
        ax2.set_title('Vertical Velocity', fontsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid(True)
        
        # Plot 3: Throttle vs Time
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(t, throttle_opt, color='green', label='Throttle', linewidth=4)
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.set_ylabel('Throttle [0-1]', fontsize=20)
        ax3.set_title('Throttle', fontsize=22)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.grid(True)
        
        # Plot 4: Propellant Mass vs Time
        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(t, mps, color='black', label='Propellant Mass', linewidth=4)
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.set_ylabel('Mass [kg]', fontsize=20)
        ax4.set_title('Propellant Mass', fontsize=22)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.grid(True)
        
        # Plot 5: Throttle Rate of Change (First Derivative)
        ax5 = plt.subplot(gs[2, 0])
        throttle_derivative = np.diff(throttle_opt) / (t[1] - t[0])
        # Add a zero at the end to match dimensions
        throttle_derivative = np.append(throttle_derivative, 0)
        ax5.plot(t, throttle_derivative, color='purple', linewidth=3)
        ax5.set_xlabel('Time [s]', fontsize=20)
        ax5.set_ylabel('Throttle Rate [1/s]', fontsize=20)
        ax5.set_title('Throttle Rate of Change', fontsize=22)
        ax5.tick_params(axis='both', which='major', labelsize=16)
        ax5.grid(True)
        
        # Plot 6: Throttle Acceleration (Second Derivative)
        ax6 = plt.subplot(gs[2, 1])
        throttle_second_derivative = np.diff(throttle_derivative) / (t[1] - t[0])
        # Add a zero at the end to match dimensions
        throttle_second_derivative = np.append(throttle_second_derivative, 0)
        ax6.plot(t, throttle_second_derivative, color='orange', linewidth=3)
        ax6.set_xlabel('Time [s]', fontsize=20)
        ax6.set_ylabel('Throttle Acceleration [1/s²]', fontsize=20)
        ax6.set_title('Throttle Acceleration', fontsize=22)
        ax6.tick_params(axis='both', which='major', labelsize=16)
        ax6.grid(True)
        
        plt.tight_layout()
        os.makedirs('results/classical_controllers', exist_ok=True)
        plt.savefig('results/classical_controllers/landing_burn_optimisation_simulation_results.png')
        plt.close()

    def save_optimisation_data(self):
        os.makedirs('data/reference_trajectory/landing_burn_controls', exist_ok=True)
        filename = 'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_history.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Objective', 'Altitude_Violation', 'Velocity_Violation',
                            'Dynamic_Pressure_Violation', 'Propellant_Violation', 'Vy_Dot_Violation', 'Burn_Time',
                            'Final_Altitude', 'Final_Velocity', 'Min_Propellant_Mass'])
            for i in range(len(self.callback.iterations)):
                writer.writerow([
                    self.callback.iterations[i],
                    self.callback.objectives[i],
                    self.callback.alt_violations[i],
                    self.callback.vel_violations[i],
                    self.callback.dyn_violations[i],
                    self.callback.prop_violations[i],
                    self.callback.vy_dot_violations[i],
                    self.callback.burn_times[i],
                    self.callback.final_alts[i],
                    self.callback.final_vels[i],
                    self.callback.min_props[i]
                ])

    def save_final_parameters(self, u_opt, T_burn_opt):
        os.makedirs('data/reference_trajectory/landing_burn_controls', exist_ok=True)
        filename = 'data/reference_trajectory/landing_burn_controls/landing_burn_optimisation_final_parameters.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Burn_Time', T_burn_opt])
            writer.writerow(['Throttle_Profile'] + list(u_opt))
            writer.writerow(['N', self.N])


if __name__ == '__main__':
    landing_burn_optimiser = LandingBurnOptimiser()
    landing_burn_optimiser()