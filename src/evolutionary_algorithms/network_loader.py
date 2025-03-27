import jax.numpy as jnp
import pandas as pd
from flax.core.frozen_dict import freeze
import numpy as np
from src.agents.functions.networks import Actor
import matplotlib.pyplot as plt
import jax

def load_pso_weights():
    results_df = pd.read_csv('results/ascent_agent/particle_subswarm_optimisation_results.csv')
    pso_row = results_df[results_df['Algorithm'] == 'Particle Subswarm Optimisation']
    if pso_row.empty:
        raise ValueError("Particle Swarm Optimisation results not found in CSV")
    
    # Extract all columns that contain weights or biases
    weight_columns = [col for col in pso_row.columns if 'weight' in col]
    bias_columns = [col for col in pso_row.columns if 'bias' in col]
    
    # Group columns by layer index
    layer_weights = {}
    layer_biases = {}
    
    # Process weight columns
    for col in weight_columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            layer_idx = int(parts[0])
            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = []
            layer_weights[layer_idx].append((col, pso_row[col].values[0]))
    
    # Process bias columns
    for col in bias_columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            layer_idx = int(parts[0])
            if layer_idx not in layer_biases:
                layer_biases[layer_idx] = []
            layer_biases[layer_idx].append((col, pso_row[col].values[0]))
    
    # Sort layer indices
    all_layer_indices = sorted(set(list(layer_weights.keys()) + list(layer_biases.keys())))
    
    # Determine network structure
    # Analyze the first layer to determine input dimension
    if 0 in layer_weights and layer_weights[0]:
        first_layer_weights = sorted(layer_weights[0], key=lambda x: int(x[0].split('_')[-1]))
        num_weights_first_layer = len(first_layer_weights)
        
        # Check if we have biases for the first layer
        if 0 in layer_biases and layer_biases[0]:
            first_layer_biases = sorted(layer_biases[0], key=lambda x: int(x[0].split('_')[-1]))
            hidden_dim = len(first_layer_biases)
        else:
            # If no biases, try to infer hidden_dim from the next layer
            for idx in all_layer_indices[1:]:
                if idx in layer_biases and layer_biases[idx]:
                    hidden_dim = len(layer_biases[idx])
                    break
            else:
                # Default if we can't determine
                hidden_dim = 10
        
        input_dim = num_weights_first_layer // hidden_dim
    else:
        # Default values if we can't determine
        input_dim = 5
        hidden_dim = 10
    
    # Find the output layer
    output_layer_idx = all_layer_indices[-1]
    if output_layer_idx in layer_biases and layer_biases[output_layer_idx]:
        output_dim = len(layer_biases[output_layer_idx])
    else:
        # Default if we can't determine
        output_dim = 3
    
    # Create params dictionary for Flax
    params = {'params': {}}
    
    # Process each layer
    for i, layer_idx in enumerate(all_layer_indices):
        # Skip layers that don't have weights
        if layer_idx not in layer_weights:
            continue
            
        # Get weights for this layer
        layer_weight_values = [w[1] for w in sorted(layer_weights[layer_idx], 
                                                   key=lambda x: int(x[0].split('_')[-1]))]
        
        # Get biases for this layer (if available)
        if layer_idx in layer_biases:
            layer_bias_values = [b[1] for b in sorted(layer_biases[layer_idx], 
                                                     key=lambda x: int(x[0].split('_')[-1]))]
        else:
            # If no biases for this layer, use zeros
            if i == 0:
                layer_bias_values = [0.0] * hidden_dim
            elif i == len(all_layer_indices) - 1:
                layer_bias_values = [0.0] * output_dim
            else:
                layer_bias_values = [0.0] * hidden_dim
        
        # Reshape weights based on layer position
        if i == 0:  # Input layer
            weights = np.array(layer_weight_values).reshape(hidden_dim, input_dim).T
        elif i == len(all_layer_indices) - 1:  # Output layer
            weights = np.array(layer_weight_values).reshape(output_dim, hidden_dim).T
        else:  # Hidden layers
            weights = np.array(layer_weight_values).reshape(hidden_dim, hidden_dim).T
        
        # Add to params
        params['params'][f'Dense_{i}'] = {
            'kernel': jnp.array(weights),
            'bias': jnp.array(layer_bias_values),
        }
    
    # Freeze the params to make them immutable (required by Flax)
    return freeze(params)

def create_and_load_network():
    # Load the parameters first to determine dimensions
    params = load_pso_weights()
    
    # The dimensions are swapped - we need to fix this
    # Our loaded params have shape (5, 10) but network expects (10, 5)
    input_dim = params['params']['Dense_0']['kernel'].shape[0]  # 5
    hidden_dim = params['params']['Dense_0']['kernel'].shape[1]  # 10
    action_dim = params['params'][f'Dense_{len(params["params"])-1}']['bias'].shape[0]  # 3
    number_of_hidden_layers = len(params['params']) - 2

    # Create the network with swapped dimensions to match our loaded params
    network = Actor(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        number_of_hidden_layers=number_of_hidden_layers
    )
    
    # Initialize the network with random parameters to get the correct structure
    key = jax.random.PRNGKey(0)
    sample_state = jnp.zeros(hidden_dim)  # Use hidden_dim as input_dim (5)
    new_params = network.init(key, sample_state)
    
    # Copy parameters with the correct shapes
    for i in range(len(params['params'])):
        layer_name = f'Dense_{i}'
        new_params['params'][layer_name]['bias'] = params['params'][layer_name]['bias']
        new_params['params'][layer_name]['kernel'] = params['params'][layer_name]['kernel']
    
    # Freeze and return
    return network, new_params, hidden_dim, number_of_hidden_layers

if __name__ == "__main__":
    network, params, hidden_dim = create_and_load_network()
    
    # Optional: Visualize network predictions for a range of inputs
    # This is useful to verify the network is working as expected
    test_range = np.linspace(-1, 1, 100)
    outputs = []
    
    for x in test_range:
        sample_state = jnp.array([x, 0.0, 0.0, 0.0, 0.0])  # Example input
        mean, _ = network.apply(params, sample_state)
        outputs.append(mean[0])  # First output dimension
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_range, outputs)
    plt.title('Network Output vs Input')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.grid(True)
    plt.savefig('network_output.png')
    plt.show()