''''
HOW TO LOAD INTO SAC

from src.agents.soft_actor_critic import SoftActorCritic as Agent
from configs.agent_config import agent_config_sac as agent_config
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.supervisory_learning.supervisory_learn import load_model

# Determine number of layers and size
params, loaded_actor_params_clean, hidden_dim, hidden_layers = load_model(flight_phase='subsonic')

agent_config['model_name'] = 'SupervisoryLearning'
agent_config['hidden_dim_actor'] = hidden_dim
agent_config['number_of_hidden_layers_actor'] = hidden_layers

agent = Agent(
    state_dim=7,
    action_dim=3,
    **agent_config)

env = env(sizing_needed_bool = False,
                       flight_phase = 'subsonic')

# Update the actor parameters with the loaded parameters
agent.actor_params = loaded_actor_params_clean

'''

import jax
import pickle
import pandas as pd
from src.agents.functions.networks import Actor
from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.supervisory.env_wrapped_supervisory import supervisory_wrapper

# Function to load the model parameters
def load_model(flight_phase='subsonic'):
    filename=f'data/agent_saves/SupervisoryLearning/{flight_phase}/supervisory_network.pkl'
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    hidden_layers = -3 # input and output (x2) layers
    for key in params:
        hidden_layers += 1
    hidden_dim = len(params[key]["bias"])

    loaded_actor_params_clean = {}
    loaded_actor_params_clean['params'] = params
    return params, loaded_actor_params_clean, hidden_dim, hidden_layers # for supervisory learning and SAC, respectively


def load_supervisory_actor(flight_phase='subsonic'):
    params, loaded_actor_params_clean, hidden_dim, hidden_layers = load_model(flight_phase=flight_phase)

    if flight_phase == 'subsonic':
        action_dim = 2
    elif flight_phase == 'supersonic':
        action_dim = 2
    else:
        raise ValueError(f'Invalid flight phase: {flight_phase}')
    
    actor = Actor(action_dim=action_dim,
                  hidden_dim=hidden_dim,
                  number_of_hidden_layers=hidden_layers)
    
    actor.params = loaded_actor_params_clean['params']

    return actor, actor.params

class Agent_Supervisory_Learnt:
    def __init__(self,
                 flight_phase='subsonic'):
        self.flight_phase = flight_phase
        self.actor, self.actor_params = load_supervisory_actor(flight_phase=flight_phase)

    def select_actions_no_stochastic(self, state):
        mean, std = self.actor.apply({'params': self.actor_params}, state)
        return mean
    
    def select_actions_stochastic(self, state):
        mean, std = self.actor.apply({'params': self.actor_params}, state)
        action = mean + std * jax.random.normal(mean.shape)
        return action
    
def plot_trajectory_supervisory(flight_phase='subsonic'):
    # read file for input normalisation values
    input_normalisation_df = pd.read_csv(f'data/agent_saves/SupervisoryLearning/{flight_phase}/input_normalisation_values_{flight_phase}.csv')
    input_normalisation_values = input_normalisation_df['normalisation_value'].tolist()
    
    env = supervisory_wrapper(input_normalisation_values = input_normalisation_values,
                                flight_phase=flight_phase)
    agent = Agent_Supervisory_Learnt(flight_phase=flight_phase)
    save_path = f'results/SupervisoryLearning/{flight_phase}/'
    universal_physics_plotter(env, agent, save_path, type='supervisory')