import torch
import torch.nn as nn

### SIMPLE NETWORK ###

class simple_network:
    def __init__(self,
                 number_of_hidden_layers = 3,
                 hidden_dim = 10,
                 output_dim = 3,
                 input_dim = 5):
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

### Neural Network Classes ###
class forces_extractor(simple_network):
    def __init__(self):
        '''
        The forces feature extractor takes in the states and outputs force in the perpendicular and parallel directions to the rocket body.
        input_dim = 9 : <x, y, vx, vy, theta, theta_dot, gamma, alpha, mass>

        3 hidden layers of 10 neurons each.

        output_dim = 2 : <force_parallel, force_perpendicular>.

        With X inputs has Y parameters : Y = X*20 + 240; so more inputs isn't that bad.
        '''
        super().__init__(number_of_hidden_layers = 3,
                         hidden_dim = 10,
                         output_dim = 2,
                         input_dim = 9)


class moment_and_force_extractor(simple_network):
    def __init__(self,
                 force_network_individual):
        super().__init__() # Inputs don't matter as just to use functions.

        self.moment_hidden_dim = 5
        self.moment_hidden_layers = 3

        # Make force actor network
        force_actor = forces_extractor().update_individiual(force_network_individual)

        # Force output and feature extractor is frozen; grads allowed as evolutionary algorithm updates instead.
        self.force_feature_extractor = nn.Sequential(*list(force_actor.network)[:-2])
        self.force_output_branch = nn.Sequential(*list(force_actor.network)[-2:])

        # Initialise network
        self.moment_output_branch = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.moment_hidden_dim, self.moment_hidden_dim), nn.ReLU()) for _ in range(self.moment_hidden_layers)],
            nn.Linear(self.moment_hidden_dim, 1),
            nn.Tanh()
        )

        # Just to be compatible with simple_network : so functions update_individiual and return_setup_vals are compatible
        self.network = self.moment_output_branch

    def forward(self, state):
        # State is <x, y, vx, vy, theta, theta_dot, gamma, alpha, mass>
        
        features = self.force_feature_extractor(state)
        force_output = self.force_output_branch(features) # <force_parallel, force_perpendicular>

        features_moment = torch.cat((features, state[7].unsqueeze(1)), dim=1) # <features, alpha>
        moment_output = self.moment_output_branch(features_moment) # <moment>
        actions = torch.cat((moment_output, force_output), dim=1) # <moment, force_parallel, force_perpendicular>
        return actions