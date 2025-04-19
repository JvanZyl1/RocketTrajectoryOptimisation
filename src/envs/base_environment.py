import csv
import pandas as pd
import numpy as np

from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.rockets_physics import compile_physics
from src.envs.rl.rtd_rl import compile_rtd_rl
from src.envs.pso.rtd_pso import compile_rtd_pso
from src.envs.supervisory.rtd_supervisory_mock import compile_rtd_supervisory_test
from src.RocketSizing.main_sizing import size_rocket

def load_supersonic_initial_state(type):
    if type == 'supervisory':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/subsonic/trajectory.csv')
    elif type == 'pso':
        data = pd.read_csv('data/pso_saves/subsonic/trajectory.csv')
    elif type == 'rl':
        data = pd.read_csv('data/agent_saves/VanillaSAC/subsonic/trajectory.csv')
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
                                      0,                                                        # y [m]
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

def load_flip_over_initial_state(type):
    if type == 'supervisory':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/supersonic/trajectory.csv')
    elif type == 'pso':
         data = pd.read_csv('data/pso_saves/supersonic/trajectory.csv')
    elif type == 'rl':
        data = pd.read_csv('data/agent_saves/VanillaSAC/supersonic/trajectory.csv')
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

def load_high_altitude_ballistic_arc_initial_state(type):
    if type == 'supervisory':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/flip_over_boostback/trajectory.csv')
    elif type == 'pso':
        data = pd.read_csv('data/pso_saves/flip_over_boostback/trajectory.csv')
    elif type == 'rl':
        data = pd.read_csv('data/agent_saves/VanillaSAC/flip_over_boostback/trajectory.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

class rocket_environment_pre_wrap:
    def __init__(self,
                 type = 'rl',
                 flight_phase = 'subsonic'):
        # Ensure state_initial is set before run_test_physics
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']
        self.flight_phase = flight_phase

        self.dt = 0.1
        if flight_phase == 'subsonic':
            self.state_initial = load_subsonic_initial_state()
        elif flight_phase == 'supersonic':
            self.state_initial = load_supersonic_initial_state(type)
        elif flight_phase == 'flip_over_boostbackburn':
            self.state_initial = load_flip_over_initial_state(type)
            self.gimbal_angle_deg = 0.0
            
        self.physics_step = compile_physics(self.dt,
                                            flight_phase=flight_phase)
        
        self.state = self.state_initial
        self.type = type

        assert type in ['rl', 'pso', 'supervisory']

        if type == 'rl':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_rl(flight_phase = flight_phase)
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
        return self.state

    def step(self, actions):
        # Physics step
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.state, info = self.physics_step(self.state,
                                                 actions)
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.gimbal_angle_deg)
            self.gimbal_angle_deg = info['action_info']['gimbal_angle_deg']

        info['state'] = self.state
        info['actions'] = actions

        truncated, self.truncation_id = self.truncated_func(self.state)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated)        

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
    

