import torch
import torch.nn as nn

from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent
from src.envs.env_endo.physics_plotter import test_agent_interaction_evolutionary_algorithms
from src.evolutionary_algorithms.env_EA import simple_actor

class forces_feature_extractor(simple_actor):
    def __init__(self):
        '''
        The forces feature extractor takes in the states and outputs force in the perpendicular and parallel directions to the rocket body.
        input_dim = 7 : <x, y, vx, vy, theta, theta_dot, mass>.
        Not including states:
        - alpha: as that is taken care of by the moment controller.
        - gamma: as should be equal to theta.
        - mass of propellant: as have current mass.
        - time: not wanted as a Markov process, which shouldn't be dependent on time.

        3 hidden layers of 10 neurons each.

        output_dim = 2 : <force_parallel, force_perpendicular>.

        With X inputs has Y parameters : Y = X*20 + 240; so more inputs isn't that bad.
        '''
        super().__init__(number_of_hidden_layers = 3,
                         hidden_dim = 10,
                         output_dim = 2,
                         input_dim = 7)
        
        # Functions are the same as simple_actor so no need to redefine them.

class MomentsAndForcesExtractor(nn.Module):
    def __init__(self,
                 force_actor,
                 moment_hidden_dim = 5):
        super(MomentsAndForcesExtractor, self).__init__()
        force_network = force_actor.network
        
        # Frozen feature extractor: all layers except the final two (linear and activation function)
        self.force_feature_extractor = nn.Sequential(*list(force_network)[:-2])
        for param in self.force_feature_extractor.parameters():
            param.requires_grad = False

        # Frozen force output branch: final two layers (linear and activation function)
        self.force_output_branch = nn.Sequential(*list(force_network)[-2:])
        for param in self.force_output_branch.parameters():
            param.requires_grad = False

        # Dynamically retrieve hidden dimension from last linear layer in the frozen extractor
        hidden_dim = None
        for layer in reversed(self.force_feature_extractor):
            if isinstance(layer, nn.Linear):
                hidden_dim = layer.out_features
                break
        if hidden_dim is None:
            raise ValueError("No linear layer found in force_feature_extractor")
            
        # Initialise network
        self.moment_output_branch = nn.Sequential(
            nn.Linear(hidden_dim, moment_hidden_dim),
            nn.ReLU(),
            nn.Linear(moment_hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, state):
        features = self.force_feature_extractor(state)
        force_output = self.force_output_branch(features) # <force_parallel, force_perpendicular>
        moment_output = self.moment_output_branch(features) # <moment>
        actions = torch.cat((moment_output, force_output), dim=1) # <moment, force_parallel, force_perpendicular>
        return actions

    def update_individiual(self, individual):
        # Update the moment output branch
        remaining_elements = individual
        
        # Assign weights and biases to each layer in the moment output branch
        param_index = 0
        for name, param in self.moment_output_branch.named_parameters():
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
        for name, param in self.moment_output_branch.named_parameters():
            flat_param = param.data.flatten().tolist()
            for j, val in enumerate(flat_param):
                mock_individual_dictionary[f'{name.replace(".", "_")}_{j}'] = val
                bounds.append((-bound_scale, bound_scale))
                param_index += 1
        
        return mock_individual_dictionary, bounds