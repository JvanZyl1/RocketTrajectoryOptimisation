import csv
import math
import numpy as np
from scipy.optimize import minimize
from src.envs.base_environment import load_re_entry_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
import matplotlib.pyplot as plt

def throttle_controller(mach_number, air_density, speed_of_sound, Q_max):
    Kp_mach = 0.043
    Q_ref = Q_max - 1000 # [Pa]
    mach_number_max = math.sqrt(2 * Q_ref / air_density) * 1 / speed_of_sound
    error_mach_number = mach_number_max - mach_number
    non_nominal_throttle = np.clip(Kp_mach * error_mach_number, 0, 1) # minimum 40% throttle
    return non_nominal_throttle

def ACS_controller(state,
                   dynamic_pressure,
                   previous_alpha_effective_rad,
                   previous_derivative,
                   max_deflection_angle_deg,
                   dt,
                   individual):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    alpha_effective_rad = gamma - theta - math.pi

    # Define gain schedules for increasing and decreasing dynamic pressure
    N_alpha_ballistic_arc = 30
    if dynamic_pressure < 5000:
        Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = individual[:2]
    elif dynamic_pressure < 10000:
        Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = individual[2:4]
    elif dynamic_pressure < 15000:
        Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = individual[4:6]
    elif dynamic_pressure < 20000:
        Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = individual[6:8]
    else: # 20000 > dynamic_pressure
        Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = individual[8:]

    delta_norm, new_derivative = PD_controller_single_step(Kp=Kp_alpha_ballistic_arc,
                                                           Kd=Kd_alpha_ballistic_arc,
                                                           N=N_alpha_ballistic_arc,
                                                           error=alpha_effective_rad,
                                                           previous_error=previous_alpha_effective_rad,
                                                           previous_derivative=previous_derivative,
                                                           dt=dt)

    # Clip control output
    delta_norm = np.clip(delta_norm, -1, 1)
    
    # Convert to degrees
    delta_left_deg = delta_norm * max_deflection_angle_deg 
    delta_right_deg = delta_left_deg

    return delta_left_deg, delta_right_deg, alpha_effective_rad, new_derivative

def augment_action_ACS(delta_left_deg, delta_right_deg, max_deflection_angle_deg):
    u0 = delta_left_deg / max_deflection_angle_deg
    u1 = delta_right_deg / max_deflection_angle_deg
    return u0, u1

def augment_action_throttle(non_nominal_throttle):
    u2 = 2 * non_nominal_throttle - 1
    return u2

class ReEntryBurnGainSchedule:
    def __init__(self, individual):
        self.dt = 0.1
        self.max_deflection_angle_deg = 60
        self.Q_max = 30000 # [Pa]
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 're_entry_burn')
        
        self.acs_controller_lambda = lambda state, dynamic_pressure, previous_alpha_effective_rad, previous_derivative: ACS_controller(state,
                                                                                                                     dynamic_pressure,
                                                                                                                     previous_alpha_effective_rad,
                                                                                                                     previous_derivative,
                                                                                                                     max_deflection_angle_deg = self.max_deflection_angle_deg,
                                                                                                                     dt = self.dt,
                                                                                                                     individual = individual)
        
        self.augment_action_ACS_lambda = lambda delta_left_deg, delta_right_deg: augment_action_ACS(delta_left_deg,
                                                                                                    delta_right_deg,
                                                                                                    max_deflection_angle_deg = self.max_deflection_angle_deg)
        
        self.throttle_controller_lambda = lambda mach_number, air_density, speed_of_sound: throttle_controller(mach_number,
                                                                                                               air_density, 
                                                                                                               speed_of_sound,
                                                                                                               self.Q_max)
        
        self.augment_action_throttle_lambda = lambda throttle: augment_action_throttle(throttle)
        
        self.initial_conditions()

    def initial_conditions(self):
        self.delta_left_deg_prev, self.delta_right_deg_prev = 0.0, 0.0
        self.state = load_re_entry_burn_initial_state('supervisory')
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

        self.y0 = self.state[1]
        self.air_density, self.atmospheric_pressure, self.speed_of_sound = endo_atmospheric_model(self.y0)
        speed = math.sqrt(self.state[2]**2 + self.state[3]**2)
        self.mach_number = speed / self.speed_of_sound
        self.dynamic_pressure = 0.5 * self.air_density * speed**2

    def reset(self, individual):
        self.initial_conditions()
        self.acs_controller_lambda = lambda state, dynamic_pressure, previous_alpha_effective_rad, previous_derivative: ACS_controller(state,
                                                                                                                     dynamic_pressure,
                                                                                                                     previous_alpha_effective_rad,
                                                                                                                     previous_derivative,
                                                                                                                     max_deflection_angle_deg = self.max_deflection_angle_deg,
                                                                                                                     dt = self.dt,
                                                                                                                     individual = individual)

    def closed_loop_step(self):
        delta_left_deg, delta_right_deg, self.previous_alpha_effective_rad, self.previous_derivative \
            = self.acs_controller_lambda(self.state, self.dynamic_pressure, self.previous_alpha_effective_rad, self.previous_derivative)
        u0, u1 = self.augment_action_ACS_lambda(delta_left_deg, delta_right_deg)
        non_nominal_throttle     = self.throttle_controller_lambda(self.mach_number, self.air_density, self.speed_of_sound)
        u2 = self.augment_action_throttle_lambda(non_nominal_throttle)
        actions = (u0, u1, u2)

        self.state, info = self.simulation_step_lambda(self.state, actions, self.delta_left_deg_prev, self.delta_right_deg_prev)
        self.delta_left_deg_prev, self.delta_right_deg_prev = info['action_info']['delta_left_deg'], info['action_info']['delta_right_deg']
        self.air_density, self.speed_of_sound, self.mach_number = info['air_density'], info['speed_of_sound'], info['mach_number']
        self.dynamic_pressure = info['dynamic_pressure']

    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        total_alpha_effective_rad = 0.0
        max_alpha_effective_rad = 0.0
        while vx < -0.1:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
            alpha_effective_rad = gamma - theta - math.pi
            total_alpha_effective_rad += abs(alpha_effective_rad)
            max_alpha_effective_rad = max(max_alpha_effective_rad, abs(alpha_effective_rad))
        return total_alpha_effective_rad, max_alpha_effective_rad, alpha_effective_rad, theta_dot
    

# Now we can run a scipy.minimise with trust-constr to find the best gains for the ACS controller
# We want to minimise the total alpha effective radians over the entire trajectory
max_alpha_effective_deg = 3.5
max_final_alpha_effective_deg = 0.05
max_final_pitch_rate_deg = 0.2

# Define the objective function
def objective_function(re_entry_burn_class, individual):
    re_entry_burn_class.reset(individual)
    total_alpha_effective_rad, max_alpha_effective_rad, _, _= re_entry_burn_class.run_closed_loop()
    return total_alpha_effective_rad

# Constraint on max alpha effective radians 
def constraint_max_alpha_effective_rad(re_entry_burn_class, individual):
    re_entry_burn_class.reset(individual)
    total_alpha_effective_rad, max_alpha_effective_rad, _, _ = re_entry_burn_class.run_closed_loop()
    return max_alpha_effective_deg - abs(math.degrees(max_alpha_effective_rad))

def constraint_final_alpha_effective_rad(re_entry_burn_class, individual):
    re_entry_burn_class.reset(individual)
    total_alpha_effective_rad, max_alpha_effective_rad, final_alpha_effective_rad, _ = re_entry_burn_class.run_closed_loop()
    return (max_final_alpha_effective_deg - abs(math.degrees(final_alpha_effective_rad)))*25

def constraint_final_pitch_rate(re_entry_burn_class, individual):
    re_entry_burn_class.reset(individual)
    total_alpha_effective_rad, max_alpha_effective_rad, final_alpha_effective_rad, final_pitch_rate_rad = re_entry_burn_class.run_closed_loop()
    return (max_final_pitch_rate_deg - abs(math.degrees(final_pitch_rate_rad)))*100

# Callback function to print iteration information and collect data
class OptimizationCallback:
    def __init__(self):
        self.iterations = []
        self.objectives = []
        self.max_alpha_violations = []
        self.final_alpha_violations = []
        self.final_pitch_rate_violations = []

    def __call__(self, xk, state=None):
        # Calculate constraint violations
        re_entry_burn_class = ReEntryBurnGainSchedule(xk)
        total_alpha_effective_rad, max_alpha_effective_rad, final_alpha_effective_rad, final_pitch_rate_rad = re_entry_burn_class.run_closed_loop()
        max_alpha_violation = max(0, max_alpha_effective_deg - math.degrees(max_alpha_effective_rad))
        final_alpha_violation = max(0, max_final_alpha_effective_deg - math.degrees(final_alpha_effective_rad))
        final_pitch_rate_violation = max(0, max_final_pitch_rate_deg - math.degrees(final_pitch_rate_rad))

        # Store data
        self.iterations.append(len(self.iterations))
        self.objectives.append(total_alpha_effective_rad)
        self.max_alpha_violations.append(max_alpha_violation)
        self.final_alpha_violations.append(final_alpha_violation)
        self.final_pitch_rate_violations.append(final_pitch_rate_violation)
        return False

# Plot optimization history
def plot_optimization_history(callback):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(callback.iterations, callback.objectives, 'b-', label='Objective')
    plt.xlabel('Iteration')
    plt.ylabel('Total Alpha Effective [rad]')
    plt.title('Objective Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(callback.iterations, callback.max_alpha_violations, 'g-', label='Max Alpha')
    plt.semilogy(callback.iterations, callback.final_alpha_violations, 'r-', label='Final Alpha')
    plt.semilogy(callback.iterations, callback.final_pitch_rate_violations, 'b-', label='Final Pitch Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/classical_controllers/re_entry_burn_optimisation_history.png')
    plt.close()

# Save optimization data
def save_optimization_data(callback, filename='data/reference_trajectory/re_entry_burn_controls/re_entry_burn_optimisation_history.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Objective', 'Max_Alpha_Violation', 'Final_Alpha_Violation',
                        'Final_Pitch_Rate_Violation', 'Final_Pitch_Rate_5000m_Violation', 'Final_Alpha_5000m_Violation'])
        for i in range(len(callback.iterations)):
            writer.writerow([
                callback.iterations[i],
                callback.objectives[i],
                callback.max_alpha_violations[i],
                callback.final_alpha_violations[i],
                callback.final_pitch_rate_violations[i]
            ])

# Modify solve_gain_schedule to use the callback
callback = OptimizationCallback()

# Solve with adjusted parameters
def solve_gain_schedule():
    # Gains can be +- 1000
    bounds = [(-50, 50),
            (-50, 50)] * 5
    # Found from only max_alpha constraint
    #x0 = [1.877e-02, 7.751e-01, 4.128e-01, -1.569e+00, -1.325e+00, 1.769e+01, 2.882e+00, 8.416e+00, 4.654e+00, -2.984e+01]
    x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    objective_func_lambda = lambda individual: objective_function(ReEntryBurnGainSchedule(x0), individual)
    constraint_func_lambda = lambda individual: constraint_max_alpha_effective_rad(ReEntryBurnGainSchedule(x0), individual)
    constraint_final_alpha_effective_rad_lambda = lambda individual: constraint_final_alpha_effective_rad(ReEntryBurnGainSchedule(x0), individual)
    constraint_final_pitch_rate_lambda = lambda individual: constraint_final_pitch_rate(ReEntryBurnGainSchedule(x0), individual)
    result = minimize(
        objective_func_lambda,
        method='trust-constr',
        bounds=bounds,
        x0 = x0,
        constraints=[{'type': 'ineq', 'fun': constraint_func_lambda},
                     {'type': 'ineq', 'fun': constraint_final_alpha_effective_rad_lambda},
                     {'type': 'ineq', 'fun': constraint_final_pitch_rate_lambda}],                     
        options={
            'maxiter': 350,
            'verbose': 2,
            'gtol': 8.2e-2,
            'xtol': 8.2e-2,
            'barrier_tol': 1e-6,
            'initial_tr_radius': 5.0
        },
        callback=callback
    )
    # Solution
    gains_ACS_re_entry_burn = result.x
    # write to csv
    with open('data/reference_trajectory/re_entry_burn_controls/ACS_re_entry_burn_gain_schedule.csv', 'w') as f:
        # titles
        f.write('Kp < 5000Pa, Kd < 5000Pa, Kp < 10000Pa, Kd < 10000Pa, Kp < 15000Pa, Kd < 15000Pa, Kp < 20000Pa, Kd < 20000Pa, Kp < 25000Pa, Kd < 25000Pa\n')
        f.write(f'{gains_ACS_re_entry_burn[0]}, {gains_ACS_re_entry_burn[1]}, {gains_ACS_re_entry_burn[2]}, {gains_ACS_re_entry_burn[3]}, {gains_ACS_re_entry_burn[4]}, {gains_ACS_re_entry_burn[5]}, {gains_ACS_re_entry_burn[6]}, {gains_ACS_re_entry_burn[7]}, {gains_ACS_re_entry_burn[8]}, {gains_ACS_re_entry_burn[9]}\n')
    # Save and plot results
    save_optimization_data(callback, 'data/reference_trajectory/re_entry_burn_controls/re_entry_burn_optimisation_history.csv')
    plot_optimization_history(callback)
    return gains_ACS_re_entry_burn