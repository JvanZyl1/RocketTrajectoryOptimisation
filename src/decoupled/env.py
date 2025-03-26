import math
import numpy as np
import torch

from decoupled.reference_trajectory import get_dt, reference_trajectory_lambda_func_y, calculate_flight_path_angles
from decoupled.physics import setup_physics
from src.decoupled.agents import forces_feature_extractor
from src.decoupled.plotter import test_agent_interaction_evolutionary_algorithms

### REWARD, TRUNCATION and DONE FUNCTIONS ###
def reward_func_zero_alpha(state, done, truncated, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    reward = 900

    # Get the reference trajectory
    xr, _, vxr, vyr, m = reference_trajectory_func(y)
    gamma_r = calculate_flight_path_angles(vxr, vyr)

    # Special errors
    if y < 0:
        return 0
    
    # are not None or Nan
    assert xr is not None and not np.isnan(xr)
    assert vxr is not None and not np.isnan(vxr)
    assert vyr is not None and not np.isnan(vyr)
    assert gamma_r is not None and not np.isnan(gamma_r)

    reward -= abs((x - xr)/xr)
    reward -= abs((vx - vxr)/vxr)
    reward -= abs((vy - vyr)/vyr)
    reward -= abs((theta - gamma_r)/gamma_r)
    reward -= abs((gamma - gamma_r)/gamma_r)
    reward += (10 - abs(math.degrees(alpha)))/10

    if y < 1000:
        reward -= 100
    # Done function
    if done:
        print(f'Done at time: {time}')
        reward += 50

    reward /= 1e6

    return reward

def truncated_func_zero_alpha(state, reference_trajectory_func):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

    # Get the reference trajectory
    xr, yr, vxr, vyr, m  = reference_trajectory_func(y)

    # Errors
    error_x = abs(x - xr)
    error_vx = abs(vx - vxr)
    error_vy = abs(vy - vyr)
    # Flight path angle (deg)
    gamma_r = calculate_flight_path_angles(vxr, vyr)
    gamma = calculate_flight_path_angles(vx, vy)
    error_gamma = abs(gamma - gamma_r)

    # If mass is depleted, return True
    if mass_propellant <= 0:
        return True
    elif error_x > 200:
        return True
    elif time > 10 and error_gamma > 20:
        return True
    elif y < -10:
        return True
    elif abs(alpha) > math.radians(10):
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
              allowable_error_x = 100,                    # [m]
              allowable_error_y = 250,                    # [m]
              allowable_error_flight_path_angle = 4):     # [deg]
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


def create_env_funcs(zero_alpha_bool = True):
    #reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
    reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y()
    if zero_alpha_bool:
        reward_func_lambda = lambda state, done, truncated : reward_func_zero_alpha(state,
                                                                                     done,
                                                                                     truncated,
                                                                                     reference_trajectory_func_y)
        truncated_func_lambda = lambda state : truncated_func_zero_alpha(state,
                                                                   reference_trajectory_func_y)
    else:
        raise NotImplementedError("Non-zero alpha physics not implemented")
    
    done_func_lambda = lambda state : done_func(state,
                                                terminal_state,
                                                allowable_error_x = 100,
                                                allowable_error_y = 100,
                                                allowable_error_flight_path_angle = 2)
    
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
                                 theta.detach(),
                                 theta_dot.detach(),
                                 alpha.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, theta, theta_dot, alpha])
    
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
                 zero_alpha_bool = True):
        # Initialise the environment
        self.env = rocket_model_wrapped(zero_alpha_bool)
        
        # Initialise the network with correct input dimension (3 for x, y, theta)
        if zero_alpha_bool:
            self.network = forces_feature_extractor()
        else:
            raise NotImplementedError("Non-zero alpha physics not implemented")
        
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