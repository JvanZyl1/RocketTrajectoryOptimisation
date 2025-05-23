import csv
import pandas as pd
import numpy as np

from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.rockets_physics import compile_physics
from src.envs.rl.rtd_rl import compile_rtd_rl
from src.envs.pso.rtd_pso import compile_rtd_pso
from src.envs.supervisory.rtd_supervisory_mock import compile_rtd_supervisory_test
from src.RocketSizing.main_sizing import size_rocket
from src.envs.disturbance_generator import compile_disturbance_generator

def load_supersonic_initial_state():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/subsonic_state_action_ascent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

def load_subsonic_initial_state():
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]
    # Initial physics state : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
    initial_physics_state = np.array([0,                                                        # x [m]
                                      1.5,                                                        # y [m] slightly up to allow for negative vy for learning.
                                      0,                                                        # vx [m/s]
                                      0,                                                        # vy [m/s]
                                      np.pi/2,                                                  # theta [rad]
                                      0,                                                        # theta_dot [rad/s]
                                      0,                                                        # gamma [rad]
                                      0,                                                        # alpha [rad]
                                      float(sizing_results['Initial mass (subrocket 0)'])*1000,             # mass [kg]
                                      float(sizing_results['Actual propellant mass stage 1'])*1000,       # mass_propellant [kg]
                                      0])                                                       # time [s]
    return initial_physics_state

def load_flip_over_initial_state():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]
    mass = last_row['mass_propellant[kg]'] + float(sizing_results['Actual structural mass stage 1'])*1000
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], mass, last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

def load_high_altitude_ballistic_arc_initial_state():
    data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state


def load_landing_burn_initial_state():
    data = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

class rocket_environment_pre_wrap:
    def __init__(self,
                 type = 'rl',
                 flight_phase = 'subsonic',
                 enable_wind = True,
                 trajectory_length = 100,
                 discount_factor = 0.99):
        # Ensure state_initial is set before run_test_physics
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS']
        self.flight_phase = flight_phase

        self.dt = 0.1
        if flight_phase == 'subsonic':
            self.state_initial = load_subsonic_initial_state()
        elif flight_phase == 'supersonic':
            self.state_initial = load_supersonic_initial_state()
        elif flight_phase == 'flip_over_boostbackburn':
            self.state_initial = load_flip_over_initial_state()
            self.gimbal_angle_deg = 0.0
        elif flight_phase == 'ballistic_arc_descent':
            self.state_initial = load_high_altitude_ballistic_arc_initial_state()
        elif flight_phase == 'landing_burn':
            self.state_initial = load_landing_burn_initial_state()
            self.gimbal_angle_deg_prev = 0.0
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        elif flight_phase == 'landing_burn_ACS':
            self.state_initial = load_landing_burn_initial_state()
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
            
        # Initialize wind generator if enabled
        self.enable_wind = enable_wind
        if enable_wind:
            self.wind_generator = compile_disturbance_generator(self.dt, flight_phase)
        else:
            self.wind_generator = None
        
        self.physics_step = compile_physics(self.dt,
                                            flight_phase=flight_phase)
        
        self.state = self.state_initial
        self.type = type

        assert type in ['rl', 'pso', 'supervisory']

        if type == 'rl':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_rl(flight_phase = flight_phase,
                                                                                   trajectory_length = trajectory_length,
                                                                                   discount_factor = discount_factor)
        elif type == 'pso':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_pso(flight_phase = flight_phase)
        elif type == 'supervisory':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_supervisory_test(flight_phase = flight_phase)

        # Startup sequence
        self.reset()
        if type == 'pso':
            self.truncation_id = 0

    def reset(self):
        self.state = self.state_initial
        if self.type == 'pso':
            self.truncation_id = 0
        if self.flight_phase == 'flip_over_boostbackburn':
            self.gimbal_angle_deg = 0.0
        elif self.flight_phase == 'landing_burn':
            self.gimbal_angle_deg_prev = 0.0
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        elif self.flight_phase == 'landing_burn_ACS':
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        if self.enable_wind:
            self.wind_generator.reset()
        return self.state

    def step(self, actions):
        # Physics step
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.state, info = self.physics_step(self.state,
                                                    actions,
                                                    wind_generator=self.wind_generator)
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.gimbal_angle_deg,
                                                 wind_generator=self.wind_generator)
            self.gimbal_angle_deg = info['action_info']['gimbal_angle_deg']
        elif self.flight_phase == 'ballistic_arc_descent':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 wind_generator=self.wind_generator)
        elif self.flight_phase == 'landing_burn':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.gimbal_angle_deg_prev,
                                                 self.delta_command_left_rad_prev,
                                                 self.delta_command_right_rad_prev,
                                                 wind_generator=self.wind_generator)
            self.gimbal_angle_deg_prev = info['action_info']['gimbal_angle_deg']
            self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
            self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']
        elif self.flight_phase == 'landing_burn_ACS':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.delta_command_left_rad_prev,
                                                 self.delta_command_right_rad_prev,
                                                 wind_generator=self.wind_generator)
            self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
            self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']
            
        info['state'] = self.state
        info['actions'] = actions
        truncated, self.truncation_id = self.truncated_func(self.state)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated, actions)
        return self.state, reward, done, truncated, info
    
    def run_test_physics(self):
        universal_physics_plotter(env = self,
                                  agent = None,
                                  save_path = f'results/physics_test/',
                                  flight_phase = self.flight_phase,
                                  type = 'physics')
        self.reset()

    def physics_step_test(self, actions, target_altitude):
        terminated = False
        self.state, info = self.physics_step(self.state, actions)
        info['state'] = self.state
        info['actions'] = actions
        altitude = self.state[1]
        propellant_mass = self.state[-2]
        if altitude >= target_altitude:
            terminated = True
        elif propellant_mass <= 0:
            terminated = True
        else:
            terminated = False

        return self.state, terminated, info
    

