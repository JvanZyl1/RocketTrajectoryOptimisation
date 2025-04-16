from src.agents.soft_actor_critic import SoftActorCritic as Agent
from configs.agent_config import agent_config_sac as agent_config
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.envs.universal_physics_plotter import universal_physics_plotter
from SupervisoryLearning.supervisory_learn import load_model

# Load the pre-trained parameters
loaded_actor_params = load_model(flight_phase='subsonic')

# Determine number of layers and size
hidden_layers = -3 # input and output (x2) layers
for key in loaded_actor_params:
    hidden_layers += 1
hidden_dim = len(loaded_actor_params[key]["bias"])

agent_config['model_name'] = 'SupervisoryLearning'
agent_config['hidden_dim_actor'] = hidden_dim
agent_config['number_of_hidden_layers_actor'] = hidden_layers

agent = Agent(
    state_dim=7,
    action_dim=3,
    **agent_config)

env = env(sizing_needed_bool = False,
                       flight_stage = 'subsonic')


loaded_actor_params_clean = {}
loaded_actor_params_clean['params'] = loaded_actor_params

for key in loaded_actor_params_clean['params']:
    assert len(agent.actor_params["params"][key]["bias"]) == len(loaded_actor_params_clean["params"][key]["bias"]), "The size of the bias is not the same"

# Update the actor parameters with the loaded parameters
agent.actor_params = loaded_actor_params_clean

universal_physics_plotter(env,
                          agent,
                          agent.save_path,
                          type = 'rl')
