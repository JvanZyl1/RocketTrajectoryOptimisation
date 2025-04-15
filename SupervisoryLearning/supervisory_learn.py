import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# Define the Actor network using flax
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 15

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        std = nn.sigmoid(nn.Dense(self.action_dim)(x))
        return mean, std

# Function to create a train state
def create_train_state(rng, learning_rate, input_dim, action_dim):
    model = Actor(action_dim=action_dim)
    params = model.init(rng, jnp.ones([1, input_dim]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Loss function
@jax.jit
def loss_fn(params, state, targets):
    mean, std = Actor(action_dim=targets.shape[1]).apply({'params': params}, state)
    loss = jnp.mean((mean - targets) ** 2)
    return loss

# Training step
@jax.jit
def train_step(state, batch):
    inputs, targets = batch
    loss, grads = jax.value_and_grad(loss_fn)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop with tqdm progress bar
def train_network(inputs, targets, epochs=25000, learning_rate=0.0001):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate, inputs.shape[1], targets.shape[1])
    losses = []
    with tqdm(range(epochs), desc='Training Progress') as pbar:
        for epoch in pbar:
            state, loss = train_step(state, (inputs, targets))
            losses.append(loss)
            # Update tqdm description every 10 epochs
            if (epoch+1) % 10 == 0:
                pbar.set_description(f'Training Progress - Loss: {loss:.6e}')
    return state, losses

# Plot learning curve with enhancements
def plot_learning_curve(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Learning Curve')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SupervisoryLearning/learning_curve.png')
    plt.show()

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    
    # Extract input features and target outputs
    inputs = data[['x', 'y', 'vx', 'vy', 'theta', 'theta_dot', 'alpha']].values
    targets = data[['MomentsApplied', 'ParallelThrust', 'PerpendicularThrust']].values

    normal_vals = np.array([np.max(data['MomentsApplied']), np.max(data['ParallelThrust']), np.max(data['PerpendicularThrust'])])

    normalised_targets = targets / normal_vals
    
    # Convert to PyTorch tensors
    inputs = jnp.array(inputs, dtype=jnp.float32)
    targets = jnp.array(normalised_targets, dtype=jnp.float32)
    
    return inputs, targets

# Function to save the model parameters
def save_model(state, filename='SupervisoryLearning/supervisory_network.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(state.params, f)
    print(f'Model parameters saved to {filename}')

# Function to load the model parameters
def load_model(filename='SupervisoryLearning/supervisory_network.pkl'):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f'Model parameters loaded from {filename}')
    return params


def test_network(batch_size=552):
    data = pd.read_csv('state_action_reference_matlab.csv')
    inputs, targets = load_data_from_csv('state_action_reference_matlab.csv') 
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    loaded_params = load_model()

    targets = data[['MomentsApplied', 'ParallelThrust', 'PerpendicularThrust']].values
    normal_vals = np.array([np.max(data['MomentsApplied']), np.max(data['ParallelThrust']), np.max(data['PerpendicularThrust'])])

    moments_applied_learnt = []
    parallel_thrust_learnt = []
    perpendicular_thrust_learnt = []
    num_batches = len(inputs) // batch_size + (1 if len(inputs) % batch_size != 0 else 0)
    for i in tqdm(range(num_batches), desc='Testing Progress'):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
        mean, std = Actor(action_dim=targets.shape[1]).apply({'params': loaded_params}, batch_inputs)
        output_values = mean
        moments_applied_learnt.extend((output_values[:, 0] * normal_vals[0]).tolist())
        parallel_thrust_learnt.extend((output_values[:, 1] * normal_vals[1]).tolist())
        perpendicular_thrust_learnt.extend((output_values[:, 2] * normal_vals[2]).tolist())

    # Plot MomentsApplied
    plt.figure(figsize=(10, 6))
    plt.plot(data['MomentsApplied'][1:-2], label='PID')
    plt.plot(moments_applied_learnt[1:-2], label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('MomentsApplied')
    plt.title('MomentsApplied Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SupervisoryLearning/MomentsApplied_Immitation.png')
    plt.close()
    # Plot ParallelThrust
    plt.figure(figsize=(10, 6))
    plt.plot(data['ParallelThrust'][1:-2], label='PID')
    plt.plot(parallel_thrust_learnt[1:-2], label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('ParallelThrust')
    plt.title('ParallelThrust Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SupervisoryLearning/ParallelThrustImitation.png')
    plt.close()
    # Plot PerpendicularThrust
    plt.figure(figsize=(10, 6))
    plt.plot(data['PerpendicularThrust'][1:-2], label='PID')
    plt.plot(perpendicular_thrust_learnt[1:-2], label='Imitation')
    plt.xlabel('Time')
    plt.ylabel('PerpendicularThrust')
    plt.title('PerpendicularThrust Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SupervisoryLearning/PerpendicularThrustImitation.png')
    plt.close()




def run_trainer():
    # Load data from CSV
    inputs, targets = load_data_from_csv('SupervisoryLearning/state_action_reference_matlab.csv')
    
    # Convert inputs and targets to JAX arrays
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    
    # Train the network
    state, losses = train_network(inputs, targets)
    
    # Plot the learning curve
    plot_learning_curve(losses)
    
    # Save the trained model
    save_model(state)
    
    # Load the model
    loaded_params = load_model()
    
    # Select 10 random indices
    random_indices = np.random.choice(len(inputs), size=10, replace=False)
    
    # Forward pass with 10 random inputs using loaded parameters
    for idx in random_indices:
        mean, _ = Actor(action_dim=targets.shape[1]).apply({'params': loaded_params}, inputs[idx].reshape(1, -1))
        output_values = mean[0]
        target_values = targets[idx]
        error_vals = []
        for i, val in enumerate(output_values):
            error_vals.append(float(abs(val - target_values[i])))
        print(f"Error: {error_vals}")


# Example usage
if __name__ == "__main__":
    run_trainer()
    test_network()