import torch.nn as nn
import numpy as np
import torch
import lmdb
import pickle
from torch.utils.tensorboard import SummaryWriter

from src.envs.base_environment import rocket_environment_pre_wrap
from src.envs.universal_physics_plotter import universal_physics_plotter
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
                 number_of_hidden_layers = 3,
                 hidden_dim = 8,
                 output_dim = 3,
                 input_dim = 7,
                 model_name = 'ascent_agent',
                 run_id = 0):
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(number_of_hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # Recalculate the number of network parameters
        self.number_of_network_parameters = sum(p.numel() for p in self.network.parameters())

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'data/pso_saves/{model_name}/runs/{run_id}')

        # Log the model graph
        dummy_input = torch.zeros((1, input_dim), dtype=torch.float32)
        self.writer.add_graph(self.network, dummy_input)
        self.writer.flush()
        self.writer.close()

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        action = self.network(state)
        return action
        
    def update_individiual(self, individual):
        # Remaining elements are flattened weights and biases
        remaining_elements = individual
        
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
        bound_scale = 0.8
        
        # Add weights and biases as flattened arrays
        param_index = 0
        for name, param in self.network.named_parameters():
            flat_param = param.data.flatten().tolist()
            for j, val in enumerate(flat_param):
                mock_individual_dictionary[f'{name.replace(".", "_")}_{j}'] = val
                bounds.append((-bound_scale, bound_scale))
                param_index += 1
        
        return mock_individual_dictionary, bounds

class pso_wrapper:
    def __init__(self,
                 sizing_needed_bool = False,
                 flight_stage = 'subsonic'):
        self.env = rocket_environment_pre_wrap(sizing_needed_bool = sizing_needed_bool,
                                               type = 'pso',
                                               flight_stage = flight_stage)
        self.initial_mass = self.env.reset()[-2]

    def truncation_id(self):
        return self.env.truncation_id

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
                                 alpha.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, vx, vy, theta, theta_dot, alpha])
    
    def step(self, action):
        action_detached = action.detach().numpy()
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state
        

class pso_wrapped_env:
    def __init__(self,
                 sizing_needed_bool = False,
                 flight_stage = 'subsonic',
                 model_name = 'ascent_agent',
                 run_id = 0):
        # Initialise the environment
        self.env = pso_wrapper(sizing_needed_bool = sizing_needed_bool,
                               flight_stage = flight_stage)
        
        # Initialise the network with correct input dimension (7 for x, y, vx, vy, theta, theta_dot, alpha)
        self.actor = simple_actor(input_dim=7,
                                  output_dim=3,
                                  model_name = model_name,
                                  run_id = run_id) # 3 actions: u0, u1, u2
        self.mock_dictionary_of_opt_params, self.bounds = self.actor.return_setup_vals()
        self.experience_buffer = []
        self.save_interval_experience_buffer = 1000       # So saves roughly every generation of a 1000 particle swarm.
        self.episode_idx = 0

        '''
        map_size is the maximum size of the database in bytes.
        1M experiences -> 57MB guess so use 100MB -> 100 * 1024 * 1024       
        2M experiences -> 114MB so use 200MB -> 200 * 1024 * 1024
        '''
        map_size = 100 * 1024 * 1024 * 2
        self.lmdb_env = lmdb.open(f'data/pso_saves/{model_name}/experience_buffer.lmdb', map_size=map_size)


    def individual_update_model(self, individual):
        self.actor.update_individiual(individual)

    def reset(self):
        _ = self.env.reset()
        self.experience_buffer = []

    def save_buffer_interval(self):
        """ Save experiences to LMDB periodically : this will be used to pre-train the critic later."""
        with self.lmdb_env.begin(write=True) as txn:
            for i, experience in enumerate(self.experience_buffer):
                txn.put(f"exp_{self.episode_idx * self.save_interval_experience_buffer + i}".encode(), pickle.dumps(experience))
        self.experience_buffer = []  # Clear buffer

    def objective_function(self, individual):
        self.individual_update_model(individual)
        state = self.env.reset()

        done_or_truncated = False
        episode_reward = 0
        previous_action = None
        previous_state = state
        previous_reward = None
        while not done_or_truncated:
            action = self.actor.forward(state)
            state, reward, done, truncated, info = self.env.step(action)
            done_or_truncated = done or truncated
            episode_reward -= reward # As minimisation problem
            if previous_action is not None:
                self.experience_buffer.append((previous_state, previous_action, previous_reward, state, action))
            previous_action = action
            previous_state = state  
            previous_reward = reward

        self.episode_idx += 1
        if self.episode_idx % self.save_interval_experience_buffer == 0 and self.episode_idx != 0:
            self.save_buffer_interval()

        return episode_reward
    
    def plot_results(self, individual, model_name):
        save_path = f'results/{model_name}/'
        self.individual_update_model(individual)
        universal_physics_plotter(self.env,
                                  self.actor,
                                  save_path,
                                  type = 'pso')