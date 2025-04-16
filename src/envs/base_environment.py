import csv
import pandas as pd
import numpy as np

from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.rockets_physics import compile_physics
from src.envs.rl.rtd_rl import compile_rtd_rl
from src.envs.pso.rtd_pso import compile_rtd_pso
from src.envs.supervisory.rtd_supervisory_mock import compile_rtd_supervisory_test
from src.RocketSizing.main_sizing import size_rocket

def load_supersonic_initial_state():
    data = pd.read_csv('data/agent_saves/SupervisoryLearning/subsonic/trajectory.csv')
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
                                      (float(sizing_results['Propellant mass stage 1 (ascent)']) \
                                       + float(sizing_results['Propellant mass stage 1 (descent)']))*1000,       # mass_propellant [kg]
                                      0])                                                       # time [s]
    return initial_physics_state

class rocket_environment_pre_wrap:
    def __init__(self,
                 sizing_needed_bool = False,
                 type = 'rl',
                 flight_phase = 'subsonic'):
        # Ensure state_initial is set before run_test_physics
        self.dt = 0.1   #get_dt()
        if flight_phase == 'subsonic':
            self.state_initial = load_subsonic_initial_state()
        elif flight_phase == 'supersonic':
            self.state_initial = load_supersonic_initial_state()
            
        self.physics_step = compile_physics(self.dt,
                                            flight_phase=flight_phase,
                                            initial_state = self.state_initial)
        
        self.state = self.state_initial
        self.type = type

        if sizing_needed_bool:
            size_rocket()
            self.run_test_physics()

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
        return self.state

    def step(self, actions):
        # Physics step
        self.state, info = self.physics_step(self.state,
                                             actions)
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
    

