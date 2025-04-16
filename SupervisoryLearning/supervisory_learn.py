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


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 50
    number_of_hidden_layers: int = 14

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
def create_train_state(rng, optimizer, input_dim, action_dim):
    model = Actor(action_dim=action_dim)
    params = model.init(rng, jnp.ones([1, input_dim]))['params']
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

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


# Plot learning curve with enhancements
def plot_learning_curve(losses, flight_phase='subsonic'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Learning Curve')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'SupervisoryLearning/results/{flight_phase}/learning_curve.png')
    plt.close()

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    
    # Extract input features and target outputs
    inputs = data[['x', 'y', 'vx', 'vy', 'theta', 'theta_dot', 'alpha', 'mass']].values
    targets = data[['MomentsApplied', 'ParallelThrust', 'PerpendicularThrust']].values

    # Normalise inputs by their absolute max values
    max_values = np.max(np.abs(inputs), axis=0)
    print(f'max_values: {max_values}')
    inputs = inputs / max_values

    normal_vals = np.array([np.max(data['MomentsApplied']), np.max(data['ParallelThrust']), np.max(data['PerpendicularThrust'])])

    normalised_targets = targets / normal_vals
    
    # Convert to PyTorch tensors
    inputs = jnp.array(inputs, dtype=jnp.float32)
    targets = jnp.array(normalised_targets, dtype=jnp.float32)
    
    return inputs, targets, max_values

# Function to save the model parameters
def save_model(state, flight_phase='subsonic'):
    filename=f'SupervisoryLearning/saves/{flight_phase}/supervisory_network.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(state.params, f)
    print(f'Model parameters saved to {filename}')

# Function to load the model parameters
def load_model(flight_phase='subsonic'):
    filename=f'SupervisoryLearning/saves/{flight_phase}/supervisory_network.pkl'
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f'Model parameters loaded from {filename}')
    return params


def test_network(flight_phase='subsonic', batch_size=552):
    data = pd.read_csv(f'SupervisoryLearning/data/state_action_reference_{flight_phase}_matlab.csv')
    inputs, targets, _ = load_data_from_csv(f'SupervisoryLearning/data/state_action_reference_{flight_phase}_matlab.csv')
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    loaded_params = load_model(flight_phase)

    targets = data[['MomentsApplied', 'ParallelThrust', 'PerpendicularThrust']].values
    max_moment = max(np.max(data['MomentsApplied']), abs(np.min(data['MomentsApplied'])))
    max_parallel_thrust = max(np.max(data['ParallelThrust']), abs(np.min(data['ParallelThrust'])))
    max_perpendicular_thrust = max(np.max(data['PerpendicularThrust']), abs(np.min(data['PerpendicularThrust'])))
    normal_vals = np.array([max_moment, max_parallel_thrust, max_perpendicular_thrust])

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
    plt.savefig(f'SupervisoryLearning/results/{flight_phase}/MomentsApplied_Immitation.png')
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
    plt.savefig(f'SupervisoryLearning/results/{flight_phase}/ParallelThrustImitation.png')
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
    plt.savefig(f'SupervisoryLearning/results/{flight_phase}/PerpendicularThrustImitation.png')
    plt.close()

def run_trainer(flight_phase='subsonic'):
    # Load data from CSV
    inputs, targets, input_normalisation_values = load_data_from_csv(f'SupervisoryLearning/data/state_action_reference_{flight_phase}_matlab.csv')
    # Write input normalisation values to a txt file
    with open(f'SupervisoryLearning/data/input_normalisation_values_{flight_phase}.txt', 'w') as f:
        f.write(f'{input_normalisation_values}')

    # Convert inputs and targets to JAX arrays
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    
    # Define the cosine decay schedule
    if flight_phase == 'subsonic':
        initial_learning_rate = 0.0001  # Set your initial learning rate
        epochs = 100000  # Total number of steps for decay
        alpha = 0.000001  # Minimum learning rate value as a fraction of initial_learning_rate
    elif flight_phase == 'supersonic':
        initial_learning_rate = 0.001  # Set your initial learning rate
        epochs = 100000  # Total number of steps for decay
        alpha = 0.0000001  # Minimum learning rate value as a fraction of initial_learning_rate

    cosine_decay_schedule = optax.cosine_decay_schedule(
        init_value=initial_learning_rate,
        decay_steps=epochs,
        alpha=alpha
    )

    # Apply the schedule to the optimizer
    actor_optimizer = optax.adam(learning_rate=cosine_decay_schedule)

    # Initialize the train state with the optimizer
    state = create_train_state(jax.random.PRNGKey(0), actor_optimizer, inputs.shape[1], targets.shape[1])

    losses = []
    with tqdm(range(epochs), desc='Training Progress') as pbar:
        for epoch in pbar:
            state, loss = train_step(state, (inputs, targets))
            losses.append(loss)
            # Update tqdm description every 10 epochs
            if loss < 6e-6:
                break
            if (epoch+1) % 100 == 0:
                pbar.set_description(f'Training Progress - Loss: {loss:.6e}')
    plot_learning_curve(losses, flight_phase)
    save_model(state, flight_phase)
    test_network(flight_phase)

# Example usage
if __name__ == "__main__":
    run_trainer(flight_phase='subsonic')