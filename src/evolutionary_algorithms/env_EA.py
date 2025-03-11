import torch.nn as nn
import numpy as np
import torch

from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent
from src.envs.env_endo.physics_plotter import test_agent_interaction_evolutionary_algorithms

'''
class model:
    def __init__(self):
    def objective_function(individual):        
    def reset(self):
    def individual_update_model(self, individual):
    def plot_results(self):
'''
class simple_actor:
    def __init__(self,
                 number_of_hidden_layers = 2,
                 hidden_dim = 5,
                 output_dim = 2,
                 input_dim = 5):
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(number_of_hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

        # Recalculate the number of network parameters
        self.number_of_network_parameters = sum(p.numel() for p in self.network.parameters())

        # Initialise the normalisation parameters
        self.state_normalisation_parameters = np.zeros(self.input_dim)
        self.action_normalisation_parameters = np.zeros(self.output_dim)

    def forward(self, state):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Convert normalization parameters to tensor
        state_norm = torch.tensor(self.state_normalisation_parameters, dtype=torch.float32)
        action_norm = torch.tensor(self.action_normalisation_parameters, dtype=torch.float32)
        
        # Apply normalization
        state = (state - state_norm) / (state_norm + 1e-8)
        
        # Forward pass through network
        action = self.network(state)
        
        # Denormalize action
        action = action / (action_norm + 1e-8)
        
        return action
        
    def update_individiual(self, individual):
        '''
        individual is structured as:
        first state normalisation parameters
        then action normalisation parameters
        then flattened weights and biases
        '''
        # First state normalisation parameters
        self.state_normalisation_parameters = individual[:self.input_dim]

        # Then action normalisation parameters
        self.action_normalisation_parameters = individual[self.input_dim:self.input_dim+self.output_dim]

        # Remaining elements are flattened weights and biases
        remaining_elements = individual[self.input_dim+self.output_dim:]
        
        # Assign weights and biases to each layer
        param_index = 0
        for name, param in self.network.named_parameters():
            param_size = param.numel()
            if param_index + param_size <= len(remaining_elements):
                param_data = torch.tensor(
                    remaining_elements[param_index:param_index+param_size], 
                    dtype=torch.float32
                ).view(param.shape)
                param.data = param_data
                param_index += param_size
            else:
                print(f"Warning: Not enough elements in individual for parameter {name}")

    def return_setup_vals(self):
        # Return a dictionary with the state normalisation parameters, action normalisation parameters, weights and biases
        mock_individual_dictionary = {}
        bounds = []
        
        # Add state normalization parameters
        for i in range(len(self.state_normalisation_parameters)):
            mock_individual_dictionary[f'state_normalisation_parameter_{i}'] = self.state_normalisation_parameters[i]
            bounds.append((-1e9, 1e9))
        
        # Add action normalization parameters
        for i in range(len(self.action_normalisation_parameters)):
            mock_individual_dictionary[f'action_normalisation_parameter_{i}'] = self.action_normalisation_parameters[i]
            bounds.append((-10, 10))
        
        # Add weights and biases as flattened arrays
        param_index = 0
        for name, param in self.network.named_parameters():
            flat_param = param.data.flatten().tolist()
            for j, val in enumerate(flat_param):
                mock_individual_dictionary[f'{name.replace(".", "_")}_{j}'] = val
                bounds.append((-1, 10))
                param_index += 1
        
        return mock_individual_dictionary, bounds

class endo_ascent_wrapped_EA:
    def __init__(self):
        self.env = rocket_model_endo_ascent()

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        
        # Handle tensors by detaching them before converting to numpy
        if isinstance(x, torch.Tensor):
            return torch.tensor([x.detach(), y.detach()], dtype=torch.float32)
        else:
            return np.array([x, y])
    
    def step(self, action):
        action_detached = action.detach().numpy()
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state
        

class env_EA_endo_ascent:
    def __init__(self):
        # Initialise the environment
        self.env = endo_ascent_wrapped_EA()
        
        # Initialise the network with correct input dimension (2 for x, y)
        self.actor = simple_actor(input_dim=2)
        self.mock_dictionary_of_opt_params, self.bounds = self.actor.return_setup_vals()

    def individual_update_model(self, individual):
        self.actor.update_individiual(individual)

    def reset(self):
        _ = self.env.reset()

    def objective_function(self, individual):
        self.individual_update_model(individual)
        state = self.env.reset()

        done_or_truncated = False
        episode_reward = 0
        while not done_or_truncated:
            action = self.actor.forward(state)
            state, reward, done, truncated, info = self.env.step(action)
            done_or_truncated = done or truncated
            episode_reward -= reward # As minimisation problem
            
        return episode_reward
    
    def plot_results(self, individual, model_name, algorithm_name):
        save_path = f'results/{model_name}/{algorithm_name}/'
        self.individual_update_model(individual)
        test_agent_interaction_evolutionary_algorithms(self,
                                                       save_path)
        
        