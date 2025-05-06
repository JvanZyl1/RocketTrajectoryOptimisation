import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyswarm import pso

from src.envs.base_environment import load_high_altitude_ballistic_arc_initial_state
from src.envs.rockets_physics import compile_physics
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.classical_controls.utils import PD_controller_single_step

# Global variables to track optimization progress
optimization_history = {
    'iterations': [],
    'best_score': [],
    'parameters': []
}

def angle_of_attack_controller(state,
                               previous_alpha_effective_rad,
                               previous_derivative,
                               dt,
                               Kp_alpha=1.2,
                               Kd_alpha=2.4,
                               N_alpha=14):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    alpha_effective_rad = gamma - theta - math.pi

    RCS_throttle, new_derivative = PD_controller_single_step(Kp=Kp_alpha,
                                                             Kd=Kd_alpha,
                                                             N=N_alpha,
                                                             error=alpha_effective_rad,
                                                             previous_error=previous_alpha_effective_rad,
                                                             previous_derivative=previous_derivative,
                                                             dt=dt)

    # Clip the result
    RCS_throttle = np.clip(RCS_throttle, -1, 1)
    return RCS_throttle, new_derivative, alpha_effective_rad

class HighAltitudeBallisticArcDescent:
    def __init__(self, individual=None):
        self.dynamic_pressure_threshold = 2000
        self.dt = 0.1
        with open('data/rocket_parameters/sizing_results.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'max_RCS_force_per_thruster':
                    max_RCS_force_per_thruster = float(row[2])
                if row[0] == 'd_base_rcs_bottom':
                    d_base_rcs_bottom = float(row[2])
                if row[0] == 'd_base_rcs_top':
                    d_base_rcs_top = float(row[2])

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'ballistic_arc_descent')

        self.load_initial_conditions()
        self.initialise_logging()
        
        # Use optimized parameters if provided, otherwise use defaults or load from file
        if individual is not None:
            self.Kp_alpha, self.Kd_alpha = individual
            self.post_process_results = False
        else:
            gains = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/gains.csv')
            self.Kp_alpha = gains['Kp_alpha'].values[0]
            self.Kd_alpha = gains['Kd_alpha'].values[0]
            self.post_process_results = True
        self.N_alpha = 1/self.dt # so no smoothing

        self.rcs_controller_lambda = lambda state, previous_alpha_effective_rad, previous_derivative: angle_of_attack_controller(
            state, 
            previous_alpha_effective_rad, 
            previous_derivative, 
            self.dt,
            self.Kp_alpha,
            self.Kd_alpha,
            self.N_alpha
        )

    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.pitch_angle_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.angle_of_attack_deg_vals = []
        self.u0_vals = []
        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.effective_angle_of_attack_deg_vals = []

    def load_initial_conditions(self):
        self.state = load_high_altitude_ballistic_arc_initial_state('supervisory')
        _, info_IC = self.simulation_step_lambda(self.state, (0.0), None)
        self.x_cog = info_IC['x_cog']
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        self.previous_alpha_effective_rad = gamma - theta - math.pi
        self.previous_derivative = 0.0

    def reset(self):
        self.state = load_high_altitude_ballistic_arc_initial_state('supervisory')
        self.initialise_logging()
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

    def closed_loop_step(self):
        RCS_throttle, self.previous_derivative, self.previous_alpha_effective_rad = self.rcs_controller_lambda(self.state,
                                                                                                               self.previous_alpha_effective_rad,
                                                                                                               self.previous_derivative)
        self.state, info = self.simulation_step_lambda(self.state, RCS_throttle, None)
        effective_angle_of_attack = self.state[6] - self.state[4] - math.pi
        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.state[4]))
        self.time_vals.append(self.state[-1])
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(info['mach_number'])
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.u0_vals.append(RCS_throttle)
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.mass_propellant_vals.append(self.state[9])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])
        self.effective_angle_of_attack_deg_vals.append(math.degrees(self.state[6] - self.state[4] - math.pi))
    def save_results(self):
        # Make sure directory exists
        save_folder = f'data/reference_trajectory/ballistic_arc_descent_controls/'
        os.makedirs(save_folder, exist_ok=True)
        
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_ballistic_arc_descent_control.csv')

        # Create a DataFrame from the collected data
        data = {
            't[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'mass[kg]': self.mass_vals
        }
        
        # Save the DataFrame to a CSV file
        pd.DataFrame(data).to_csv(full_trajectory_path, index=False)

        pitch_angle_rad_vals = np.array(self.pitch_angle_deg_vals) * np.pi / 180
        pitch_rate_rad_vals = np.array(self.pitch_rate_deg_vals) * np.pi / 180
        angle_of_attack_rad_vals = np.array(self.angle_of_attack_deg_vals) * np.pi / 180
        flight_path_angle_rad_vals = np.array(self.flight_path_angle_deg_vals) * np.pi / 180
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': pitch_angle_rad_vals,
            'theta_dot[rad/s]': pitch_rate_rad_vals,
            'alpha[rad]': angle_of_attack_rad_vals,
            'gamma[rad]': flight_path_angle_rad_vals,
            'mass[kg]': self.mass_vals,
            'u0': self.u0_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_ballistic_arc_descent_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        effective_pitch = np.array(self.pitch_angle_deg_vals) + 180
        # A4 size plot
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
        plt.suptitle('Ballistic Arc Descent Control', fontsize = 32)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(np.array(self.x_vals)/1000, np.array(self.y_vals)/1000, linewidth = 4, color = 'blue')
        ax1.set_xlabel('x [km]', fontsize = 20)
        ax1.set_ylabel('y [km]', fontsize = 20)
        ax1.set_title('Flight Path', fontsize = 22)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid()

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, np.array(self.vy_vals)/1000, linewidth = 4, color = 'blue')
        ax2.set_xlabel('Time [s]', fontsize = 20)
        ax2.set_ylabel('Velocity [km/s]', fontsize = 20)
        ax2.set_title('Vertical Velocity', fontsize = 22)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid()

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, effective_pitch, linewidth = 4, label = 'Pitch (Down))', color = 'blue')
        ax3.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 4, label = 'Flight path', color = 'red', linestyle = '--')
        ax3.set_xlabel('Time [s]', fontsize = 20)
        ax3.set_ylabel('Angle [deg]', fontsize = 20)
        ax3.set_title('Pitch and Flight Path Angles', fontsize = 22)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.legend(fontsize = 20)
        ax3.grid()

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.u0_vals, linewidth = 4, label = 'RCS Throttle', color = 'blue')
        ax4.set_xlabel('Time [s]', fontsize = 20)
        ax4.set_ylabel('Throttle [-]', fontsize = 20)
        ax4.set_title('RCS Throttle', fontsize = 22)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.grid()

        plt.savefig(f'results/classical_controllers/ballistic_arc_descent.png')
        plt.close()
    
    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        dynamic_pressure = 0.5 * density * math.sqrt(vx**2 + vy**2)**2
        while dynamic_pressure < self.dynamic_pressure_threshold:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
            density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            dynamic_pressure = 0.5 * density * math.sqrt(vx**2 + vy**2)**2
            if time > 1000:
                break
        
        if self.post_process_results:
            self.plot_results()
            self.save_results()
            
    def performance_metrics(self):
        # Calculate performance metric - minimize effective angle of attack
        reward = 0
        for alpha_eff in self.effective_angle_of_attack_deg_vals:
            reward -= abs(alpha_eff)/1e2
        # Penalize last 20% of trajectory final effective angle of attack more heavily
        last_20_percent_index = int(len(self.effective_angle_of_attack_deg_vals) * 0.8)
        # Convert list to numpy array for proper handling of abs with lists
        last_segment = np.array(self.effective_angle_of_attack_deg_vals[last_20_percent_index:])
        reward -= np.sum(np.abs(last_segment))*50
        return reward

def objective_func_lambda(individual):
    ballistic_arc = HighAltitudeBallisticArcDescent(individual)
    ballistic_arc.run_closed_loop()
    obj = -ballistic_arc.performance_metrics()
    
    # Track optimization progress
    if hasattr(objective_func_lambda, 'iteration'):
        objective_func_lambda.iteration += 1
    else:
        objective_func_lambda.iteration = 1
    
    # Store current best result if this is a new best
    if not optimization_history['best_score'] or obj < min(optimization_history['best_score']):
        optimization_history['iterations'].append(objective_func_lambda.iteration)
        optimization_history['best_score'].append(obj)
        optimization_history['parameters'].append(individual.copy())
    
    return obj

def save_gains(xopt):
    xopt = np.array(xopt)
    print("Optimal parameters:", xopt)
    # Creating DataFrame
    gains = pd.DataFrame({'Kp_alpha': [xopt[0]], 'Kd_alpha': [xopt[1]]})
    # Make sure directory exists
    os.makedirs('data/reference_trajectory/ballistic_arc_descent_controls/', exist_ok=True)
    gains.to_csv('data/reference_trajectory/ballistic_arc_descent_controls/gains.csv', index=False)

def plot_optimization_progress():
    if not optimization_history['iterations']:
        print("No optimization history to plot")
        return
    
    # Plot the optimization progress
    plt.figure(figsize=(12, 8))
    plt.plot(optimization_history['iterations'], optimization_history['best_score'], 'o-', linewidth=4)
    plt.title('Optimization Progress - Ballistic Arc Descent Controller', fontsize=24)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Best Objective Value (lower is better)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)
    
    # Create annotation for the final parameters
    best_idx = optimization_history['best_score'].index(min(optimization_history['best_score']))
    best_params = optimization_history['parameters'][best_idx]
    best_score = optimization_history['best_score'][best_idx]

    # Ensure directory exists
    os.makedirs('results/classical_controllers/', exist_ok=True)
    plt.savefig('results/classical_controllers/ballistic_arc_optimization_progress.png')
    plt.close()
    
    # Plot parameter convergence
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param_history = np.array(optimization_history['parameters'])
    
    ax.plot(optimization_history['iterations'], param_history[:, 0], 'g-', linewidth=4, label='Kp_alpha')
    ax.plot(optimization_history['iterations'], param_history[:, 1], 'm-', linewidth=4, label='Kd_alpha')
    
    ax.set_title('Parameter Convergence - Ballistic Arc Descent Controller', fontsize=24)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Parameter Value', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)
    
    plt.savefig('results/classical_controllers/ballistic_arc_parameter_convergence.png')
    plt.close()

def tune_ballistic_arc_descent():
    # Reset the optimization history
    global optimization_history
    optimization_history = {
        'iterations': [],
        'best_score': [],
        'parameters': []
    }
    
    # Reset the objective function iteration counter
    if hasattr(objective_func_lambda, 'iteration'):
        delattr(objective_func_lambda, 'iteration')
    
    # Kp_alpha, Kd_alpha
    lb = [0.0, 0.0]  # Lower bounds
    ub = [5.0, 5.0]  # Upper bounds
    
    xopt, fopt = pso(
        objective_func_lambda,
        lb, 
        ub,
        swarmsize=40,      # Number of particles
        omega=0.5,         # Particle velocity scaling factor
        phip=0.5,          # Scaling factor for particle's best known position
        phig=0.5,          # Scaling factor for swarm's best known position
        maxiter=10,        # Maximum iterations
        minstep=1e-6,      # Minimum step size before search termination
        minfunc=1e-6,      # Minimum change in obj value before termination
        debug=True,        # Print progress statements
    )
    
    # Plot the optimization progress
    plot_optimization_progress()
    
    save_gains(xopt)
    return HighAltitudeBallisticArcDescent(xopt)

class BallisticArcDescentTuning:
    def __init__(self, tune_bool=False):
        if tune_bool:
            self.env = tune_ballistic_arc_descent()
        else:
            self.env = HighAltitudeBallisticArcDescent()

    def run_closed_loop(self):
        self.env.run_closed_loop()

    def plot_results(self):
        self.env.plot_results()

    def save_results(self):
        self.env.save_results()