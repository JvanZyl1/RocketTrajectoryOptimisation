import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

from src.agents.functions.networks import Actor
from src.envs.supervisory.agent_load_supervisory import plot_trajectory_supervisory


def loss_fn(params, state, targets, hidden_dim, number_of_hidden_layers):
    mean, std = Actor(action_dim=targets.shape[1],
                      hidden_dim=hidden_dim,
                      number_of_hidden_layers=number_of_hidden_layers).apply({'params': params}, state)
    loss = jnp.mean((mean - targets) ** 2)
    return loss

def train_step(state, batch, loss_fcn_lambda):
    inputs, targets = batch
    loss, grads = jax.value_and_grad(loss_fcn_lambda)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss

class SupervisoryLearning:
    def __init__(self,
                 flight_phase : str = 'subsonic'):
        assert flight_phase in ['subsonic', 'supersonic'], 'Flight phase must be either subsonic or supersonic'
        self.flight_phase = flight_phase


        self.inputs, self.targets, self.input_normalisation_values, self.output_normalisation_values = self.load_data_from_csv()

        # Define the cosine decay schedule
        if flight_phase == 'subsonic':
            self.epochs = 25000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.0001,
                                                    epochs = self.epochs,
                                                    alpha = 0.000001)
            self.hidden_dim = 20
            self.number_of_hidden_layers = 14

        elif flight_phase == 'supersonic':
            self.epochs = 100000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.001,
                                                    epochs = self.epochs,
                                                    alpha = 0.0000001)
            self.hidden_dim = 50
            self.number_of_hidden_layers = 14

        # Initialize the training state with the Actor model and optimizer
        self.model = Actor(action_dim=self.targets.shape[1],
                           hidden_dim=self.hidden_dim,
                           number_of_hidden_layers=self.number_of_hidden_layers)
        params = self.model.init(jax.random.PRNGKey(0), jnp.ones([1, self.inputs.shape[1]]))['params']
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                              params=params,
                                              tx=actor_optimiser)
        
        loss_fcn_lambda = jax.jit(
              partial(loss_fn,
                      hidden_dim=self.hidden_dim,
                      number_of_hidden_layers=self.number_of_hidden_layers),
              static_argnames = ['hidden_dim', 'number_of_hidden_layers']
        )

        self.train_step_lambda = jax.jit(
            partial(train_step,
                    loss_fcn_lambda=loss_fcn_lambda),
            static_argnames = ['loss_fcn_lambda']
        )

        # Logging
        self.losses = []

    def __call__(self):
        self.train()
    
    def reset(self):
        self.losses = []
        params = self.model.init(jax.random.PRNGKey(0), jnp.ones([1, self.inputs.shape[1]]))['params']
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                              params=params,
                                              tx=self.actor_optimiser)

    def train(self):
        with tqdm(range(self.epochs), desc='Training Progress') as pbar:
            for epoch in pbar:
                self.state, loss = self.train_step_lambda(self.state, (self.inputs, self.targets))
                self.losses.append(loss)
                # Update tqdm description every 10 epochs
                if loss < 6e-6:
                    break
                if (epoch+1) % 100 == 0:
                    pbar.set_description(f'Training Progress - Loss: {loss:.6e}')

        self.plot_learning_curve(self.losses)
        self.save_model(self.state)
        self.test_network()


    def create_optimiser(self,
                         initial_learning_rate : float = 0.0001,
                         epochs : int = 100000,
                         alpha : float = 0.000001):
        cosine_decay_schedule = optax.cosine_decay_schedule(
            init_value=initial_learning_rate,
            decay_steps=epochs,
            alpha=alpha
            )
        return optax.adam(learning_rate=cosine_decay_schedule)
    
    def load_data_from_csv(self):
        self.reference_data = pd.read_csv(f'data/reference_trajectory/MATLAB/state_action_reference_{self.flight_phase}_matlab.csv')
        
        # Extract input features and target outputs
        inputs = self.reference_data[['x', 'y', 'vx', 'vy', 'theta', 'theta_dot', 'alpha', 'mass']].values
        targets = self.reference_data[['MomentsApplied', 'ParallelThrust', 'PerpendicularThrust']].values

        # Normalise inputs by their absolute max values
        input_normalisation_vals = np.max(np.abs(inputs), axis=0)
        inputs = inputs / input_normalisation_vals
        with open(f'data/agent_saves/SupervisoryLearning/{self.flight_phase}/input_normalisation_values_{self.flight_phase}.txt', 'w') as f:
            f.write(f'{input_normalisation_vals}')

        output_normalisation_vals = np.array([np.max(self.reference_data['MomentsApplied']), np.max(self.reference_data['ParallelThrust']), np.max(self.reference_data['PerpendicularThrust'])])
        normalised_targets = targets / output_normalisation_vals
        with open(f'data/agent_saves/SupervisoryLearning/{self.flight_phase}/output_normalisation_values_{self.flight_phase}.txt', 'w') as f:
            f.write(f'{output_normalisation_vals}')
        
        # Convert to PyTorch tensors
        inputs = jnp.array(inputs, dtype=jnp.float32)
        targets = jnp.array(normalised_targets, dtype=jnp.float32)
        
        return inputs, targets, input_normalisation_vals, output_normalisation_vals
    
    def save_model(self, state):
        filename=f'data/agent_saves/SupervisoryLearning/{self.flight_phase}/supervisory_network.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(state.params, f)
        print(f'Model parameters saved to {filename}')

    def plot_learning_curve(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (log scale)')
        plt.title('Learning Curve')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/SupervisoryLearning/{self.flight_phase}/learning_curve.png')
        plt.close()

    def test_network(self):
        batch_size_tester = 552
        moments_applied_learnt = []
        parallel_thrust_learnt = []
        perpendicular_thrust_learnt = []
        num_batches = len(self.inputs) // batch_size_tester + (1 if len(self.inputs) % batch_size_tester != 0 else 0)
        for i in tqdm(range(num_batches), desc='Testing Progress'):
            batch_inputs = self.inputs[i*batch_size_tester:(i+1)*batch_size_tester]
            mean, std = Actor(action_dim=self.targets.shape[1],
                              hidden_dim=self.hidden_dim,
                              number_of_hidden_layers=self.number_of_hidden_layers).apply({'params': self.state.params}, batch_inputs)
            output_values = mean
            moments_applied_learnt.extend((output_values[:, 0] * self.output_normalisation_values[0]).tolist())
            parallel_thrust_learnt.extend((output_values[:, 1] * self.output_normalisation_values[1]).tolist())
            perpendicular_thrust_learnt.extend((output_values[:, 2] * self.output_normalisation_values[2]).tolist())
    
        # Plot MomentsApplied
        plt.figure(figsize=(10, 6))
        plt.plot(self.reference_data['MomentsApplied'][1:-2], label='PID')
        plt.plot(moments_applied_learnt[1:-2], label='Imitation')
        plt.xlabel('Time')
        plt.ylabel('MomentsApplied')
        plt.title('MomentsApplied Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/SupervisoryLearning/{self.flight_phase}/MomentsApplied_Immitation.png')
        plt.close()
        # Plot ParallelThrust
        plt.figure(figsize=(10, 6))
        plt.plot(self.reference_data['ParallelThrust'][1:-2], label='PID')
        plt.plot(parallel_thrust_learnt[1:-2], label='Imitation')
        plt.xlabel('Time')
        plt.ylabel('ParallelThrust')
        plt.title('ParallelThrust Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/SupervisoryLearning/{self.flight_phase}/ParallelThrustImitation.png')
        plt.close()
        # Plot PerpendicularThrust
        plt.figure(figsize=(10, 6))
        plt.plot(self.reference_data['PerpendicularThrust'][1:-2], label='PID')
        plt.plot(perpendicular_thrust_learnt[1:-2], label='Imitation')
        plt.xlabel('Time')
        plt.ylabel('PerpendicularThrust')
        plt.title('PerpendicularThrust Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/SupervisoryLearning/{self.flight_phase}/PerpendicularThrustImitation.png')
        plt.close()

        plot_trajectory_supervisory(self.input_normalisation_values,
                                    self.flight_phase)