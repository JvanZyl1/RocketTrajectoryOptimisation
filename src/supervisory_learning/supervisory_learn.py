import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

from src.agents.functions.networks import ClassicalActor as Actor
from src.envs.utils.input_normalisation import find_input_normalisation_vals
from src.envs.supervisory.agent_load_supervisory import plot_trajectory_supervisory
from src.supervisory_learning.supervisory_test import (endo_ascent_supervisory_test, flip_over_boostbackburn_supervisory_test,\
                                                        ballistic_arc_descent_supervisory_test, re_entry_burn_supervisory_test)

def loss_fn(params, state, targets, hidden_dim, number_of_hidden_layers):
    mean = Actor(action_dim=targets.shape[1],
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
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn'], 'Flight phase must be either subsonic or supersonic or flip_over_boostbackburn or ballistic_arc_descent  '
        self.flight_phase = flight_phase

        self.inputs, self.targets, self.input_normalisation_values = self.load_data_from_csv()

        # Define the cosine decay schedule
        if flight_phase == 'subsonic':
            self.epochs = 25000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.0001,
                                                    epochs = self.epochs,
                                                    alpha = 0.000001)
            self.hidden_dim = 256
            self.number_of_hidden_layers = 5

        elif flight_phase == 'supersonic':
            self.epochs = 25000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.001,
                                                    epochs = self.epochs,
                                                    alpha = 0.0000001)
            self.hidden_dim = 256
            self.number_of_hidden_layers = 5
        elif flight_phase == 'flip_over_boostbackburn':
            self.epochs =20000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.001,
                                                    epochs = self.epochs,
                                                    alpha = 0.0000001)
            self.hidden_dim = 256
            self.number_of_hidden_layers = 3
        elif flight_phase == 'ballistic_arc_descent':
            self.epochs = 20000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.0001,
                                                    epochs = self.epochs,
                                                    alpha = 0.0000001)
            self.hidden_dim = 256
            self.number_of_hidden_layers = 4
        elif flight_phase == 're_entry_burn':
            self.epochs = 20000
            actor_optimiser = self.create_optimiser(initial_learning_rate = 0.0001,
                                                    epochs = self.epochs,
                                                    alpha = 0.0000001)
            self.hidden_dim = 256
            self.number_of_hidden_layers = 5
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
                if (epoch+1) % 100 == 0:
                    pbar.set_description(f'Training Progress - Loss: {loss:.6e}')
                if self.flight_phase == 'flip_over_boostbackburn':
                    if loss < 1e-4:
                        break
                elif self.flight_phase == 'ballistic_arc_descent':
                    if loss < 5e-6:
                        break
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
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.reference_data = pd.read_csv(f'data/reference_trajectory/ascent_controls/{self.flight_phase}_state_action_ascent_control.csv')
            inputs = self.reference_data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values[1:-2]
            targets = self.reference_data[['u0', 'u1']].values[1:-2]
            
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.reference_data = pd.read_csv(f'data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
            inputs = self.reference_data[['theta[rad]', 'theta_dot[rad/s]']]
            targets = self.reference_data[['u0']]
        elif self.flight_phase == 'ballistic_arc_descent':
            self.reference_data = pd.read_csv(f'data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
            inputs = self.reference_data[['theta[rad]', 'theta_dot[rad/s]', 'gamma[rad]', 'alpha[rad]']]
            targets = self.reference_data[['u0']]
        elif self.flight_phase == 're_entry_burn':
            self.reference_data = pd.read_csv(f'data/reference_trajectory/re_entry_burn_controls/state_action_re_entry_burn_control.csv')
            inputs = self.reference_data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'gamma[rad]', 'alpha[rad]', 'mass[kg]']]
            targets = self.reference_data[['u0', 'u1']]

        # Normalise inputs by their absolute max values
        input_normalisation_vals = find_input_normalisation_vals(self.flight_phase)
        inputs = inputs / input_normalisation_vals
        
        inputs = jnp.array(inputs, dtype=jnp.float32)
        targets = jnp.array(targets, dtype=jnp.float32)
        return inputs, targets, input_normalisation_vals
    
    def save_model(self, state):
        filename=f'data/agent_saves/SupervisoryLearning/{self.flight_phase}/supervisory_network.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(state.params, f)
        print(f'Model parameters saved to {filename}')

    def plot_learning_curve(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss (log)', fontsize=20)
        plt.title('Learning Curve', fontsize=22)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()
        plt.savefig(f'results/SupervisoryLearning/{self.flight_phase}/learning_curve.png')
        plt.close()

    def test_network(self):
        print(f'Testing network for flight phase: {self.flight_phase}')
        if self.flight_phase in ['subsonic', 'supersonic']:
            endo_ascent_supervisory_test(inputs = self.inputs,
                                         flight_phase = self.flight_phase,
                                         state_network = self.state,
                                         targets = self.targets,
                                         hidden_dim = self.hidden_dim,
                                         number_of_hidden_layers = self.number_of_hidden_layers,
                                         reference_data = self.reference_data)
        elif self.flight_phase == 'flip_over_boostbackburn':
            flip_over_boostbackburn_supervisory_test(inputs = self.inputs,
                                                     flight_phase = self.flight_phase,
                                                     state_network = self.state,
                                                     targets = self.targets,
                                                     hidden_dim = self.hidden_dim,
                                                     number_of_hidden_layers = self.number_of_hidden_layers,
                                                     reference_data = self.reference_data) 
        elif self.flight_phase == 'ballistic_arc_descent':
            ballistic_arc_descent_supervisory_test(inputs = self.inputs,
                                                    flight_phase = self.flight_phase,
                                                    state_network = self.state,
                                                    targets = self.targets,
                                                    hidden_dim = self.hidden_dim,
                                                    number_of_hidden_layers = self.number_of_hidden_layers,
                                                    reference_data = self.reference_data)
        elif self.flight_phase == 're_entry_burn':
            re_entry_burn_supervisory_test(inputs = self.inputs,
                                          flight_phase = self.flight_phase,
                                          state_network = self.state,
                                          targets = self.targets,
                                          hidden_dim = self.hidden_dim,
                                          number_of_hidden_layers = self.number_of_hidden_layers,
                                          reference_data = self.reference_data)
        else:
            raise ValueError(f'Flight phase {self.flight_phase} not supported')
        plot_trajectory_supervisory(self.flight_phase)