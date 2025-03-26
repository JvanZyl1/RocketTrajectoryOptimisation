import math
import torch
import numpy as np

from src.NetworkFitting.utils.physics import setup_physics
from src.NetworkFitting.agents import forces_extractor, moment_and_force_extractor
from src.NetworkFitting.utils.plotter import test_agent_interaction_evolutionary_algorithms
from src.NetworkFitting.reference_trajectory import get_dt, reference_trajectory_lambda_func_y, calculate_flight_path_angles

### REWARD, TRUNCATION and DONE FUNCTIONS ###
def reward_func_zero_alpha(state, done, truncated, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    reward = 900

    # Get the reference trajectory
    xr, _, vxr, vyr, _ = reference_trajectory_func(y)

    reward -= abs((x - xr)/xr)
    reward -= abs((vx - vxr)/vxr)
    reward -= abs((vy - vyr)/vyr)

    # Done function
    if done:
        print(f'Done at time: {time}')
        reward += 50000

    reward /= 1e6

    return reward

def truncated_func_zero_alpha(state, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Get the reference trajectory
    xr, _, vxr, vyr, _  = reference_trajectory_func(y)

    # Errors
    error_x = abs(x - xr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)

    # If mass is depleted, return True
    if mass_propellant <= 0:
        return True
    elif error_x > 200:
        return True
    elif y > 20000:
        if error_vx > 40:
            return True
        elif error_vy > 40:
            return True
        else:
            return False
    elif y < 20000:
        if error_vx > 20:
            return True
        elif error_vy > 20:
            return True
        else:
            return False
    else:
        return False


def done_func(state,
              terminal_state,
              allowable_error_x,                    # [m]
              allowable_error_y,                    # [m]
              allowable_error_flight_path_angle):     # [deg]
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    xr, yr, vxr, vyr, m = terminal_state
    gamma_terminal = calculate_flight_path_angles(vxr, vyr)

    # Check if mass is depleted : should be truncated
    if mass_propellant >= 0 and \
        abs(x - xr) <= allowable_error_x and \
        abs(y - yr) <= allowable_error_y and \
        abs(math.degrees(gamma) - gamma_terminal) <= allowable_error_flight_path_angle and \
            abs(alpha) <= math.radians(5):
            return True
    else:
        return False
    

def reward_func_non_zero_alpha(state, done, truncated):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    reward = 90 - abs(math.degrees(alpha))

    if done:
        print(f'Done at time: {time}')
        reward += 50

    reward /= 1e4

    return reward

def truncated_func_non_zero_alpha(state):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    if alpha > math.radians(45):
        return True
    else:
        return False


def create_env_funcs(zero_alpha_bool = True):
    reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y()
    if zero_alpha_bool:
        reward_func_lambda = lambda state, done, truncated : reward_func_zero_alpha(state,
                                                                         done,
                                                                         truncated,
                                                                         reference_trajectory_func_y)
        truncated_func_lambda = lambda state : truncated_func_zero_alpha(state,
                                                                         reference_trajectory_func_y)
    else:
        reward_func_lambda = lambda state, done, truncated : reward_func_non_zero_alpha(state,
                                                                             done,
                                                                             truncated)
        truncated_func_lambda = lambda state : truncated_func_non_zero_alpha(state)
        
    done_func_lambda = lambda state : done_func(state,
                                                terminal_state,
                                                allowable_error_x = 250,
                                                allowable_error_y = 250,
                                                allowable_error_flight_path_angle = 4)
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda


### ENVIRONMENT: GYMNASIUM styled environment ###
class rocket_model:
    def __init__(self,
                 zero_alpha_bool = True):

        self.dt = get_dt()
        self.reward_func, self.truncated_func, self.done_func = create_env_funcs(zero_alpha_bool)

        # Startup sequence
        self.physics_step, self.state_initial = setup_physics(self.dt,
                                                              zero_alpha_bool)
        self.state = self.state_initial
        self.reset()

    def reset(self):
        self.state = self.state_initial
        return self.state

    def step(self, actions):
        # Physics step
        self.state, info = self.physics_step(self.state,
                                             actions)
        info['state'] = self.state
        info['actions'] = actions

        truncated = self.truncated_func(self.state)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated)        

        return self.state, reward, done, truncated, info

    def physics_step_test(self, actions, target_altitude):
        
        terminated = False
        self.state, info = self.physics_step(self.state,
                                             actions)
        altitude = self.state[1]
        propellant_mass = self.state[-2]
        if altitude >= target_altitude:
            terminated = True
        elif propellant_mass <= 0:
            terminated = True
        else:
            terminated = False

        return self.state, terminated, info
    

### WRAPPED ENVIRONMENT : to work with a neural network###
class rocket_model_wrapped:
    def __init__(self,
                 zero_alpha_bool = True):
        self.env = rocket_model(zero_alpha_bool)
        self.initial_mass = self.env.reset()[-2]

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        
        # Handle tensors by detaching them before converting to numpy
        if isinstance(x, torch.Tensor):
            return torch.tensor([x.detach(),
                                 y.detach(),
                                 vx.detach(),
                                 vy.detach(),
                                 theta.detach(),
                                 theta_dot.detach(),
                                 gamma.detach(),
                                 alpha.detach(),
                                 mass.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, vx, vy, theta, theta_dot, gamma, alpha, mass])
    
    def step(self, action):
        action_detached = action.detach().numpy()
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state
        

### EVOLUTIONARY ALGORITHM WRAPPED ENVIRONMENT ###

class env_ea:
    def __init__(self,
                 zero_alpha_bool = True,
                 force_network_individual = None):
        # Initialise the environment
        self.env = rocket_model_wrapped(zero_alpha_bool)
        
        # Initialise the network with correct input dimension (3 for x, y, theta)
        if zero_alpha_bool:
            self.network = forces_extractor()
        else:
            assert force_network_individual is not None, "force_network_individual must be provided for non-zero alpha"
            self.network = moment_and_force_extractor(force_network_individual)
        
        self.mock_dictionary_of_opt_params, self.bounds = self.network.return_setup_vals()

    def individual_update_model(self, individual):
        self.network.update_individiual(individual)

    def reset(self):
        _ = self.env.reset()

    def objective_function(self, individual):
        self.individual_update_model(individual)
        state = self.env.reset()

        done_or_truncated = False
        episode_reward = 0
        while not done_or_truncated:
            action = self.network.forward(state)
            state, reward, done, truncated, info = self.env.step(action)
            done_or_truncated = done or truncated
            episode_reward -= reward # As minimisation problem
            
        return episode_reward
    
    def plot_results(self, individual, model_name, algorithm_name):
        save_path = f'results/{model_name}/{algorithm_name}/'
        self.individual_update_model(individual)
        test_agent_interaction_evolutionary_algorithms(self,
                                                       save_path)