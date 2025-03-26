import os

from src.trainers.trainers import Trainer_MARL
from src.envs.env_ascent import ascent_wrapped_env as env
from src.envs.env_endo.physics_plotter import test_agent_interaction
from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.functions.load_agent import load_sac


class TrainerEndo(Trainer_MARL):
    def __init__(self,
                 env,
                 worker_agent,
                 central_agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 number_of_agents: int = 2,
                 info: str = ""):
        super(TrainerEndo, self).__init__(env, worker_agent, central_agent, num_episodes, save_interval, number_of_agents, info)

    def test_env(self):
        test_agent_interaction(self.env,
                               self.central_agent)

class VerticalRisingTrain:
    def __init__(self,
                 num_episodes : int,
                 worker_agent_config : dict,
                 central_agent_config : dict,
                 debug_mode : bool = False,
                 save_interval : int = 10,
                 number_of_agents : int = 2,
                 info : str = "",
                 marl_load_info : str = None):
        self.env = env(sizing_needed_bool = False)
        self.model_name = 'VerticalRising-MARL'
        if marl_load_info is not None:
            self.load_agents(marl_load_info)
        else:
            worker_agent_config['model_name'] = self.model_name
            worker_agent_config['print_bool'] = debug_mode

            worker_agent_clone = Agent(
                seed = 0,
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                **worker_agent_config)
            
            central_agent_config['model_name'] = self.model_name
            central_agent_config['print_bool'] = debug_mode
            
            central_agent = Agent(
                seed = 0,
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                **central_agent_config)
            
            self.trainer = TrainerEndo(self.env,
                                            worker_agent_clone,
                                            central_agent,
                                            num_episodes,
                                            save_interval,
                                            number_of_agents,
                                            info)
            
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.number_of_agents = number_of_agents
        self.info = info
    
    def load_agents(self, info : str):
        # Load central agent
        central_agent = load_sac(f'data/agent_saves/{self.model_name}/saves/soft-actor-critic_{info}.pkl')

        # Load worker agents
        worker_agents = []
        base_path = f'data/agent_saves/{self.model_name}/saves/'
        i = 0
        while True:
            worker_path = os.path.join(base_path, f'soft-actor-critic_{info}_worker_{i}.pkl')
            if not os.path.exists(worker_path):
                break
            worker_agents.append(load_sac(worker_path))
            i += 1

        # Update trainer with loaded agents
        self.trainer = TrainerEndo(self.env,
                                       worker_agents[0],
                                       central_agent,
                                       self.num_episodes,
                                       self.save_interval,
                                       self.number_of_agents,
                                       self.info)
        self.trainer.load_all(central_agent, worker_agents)

    def save_all(self):
        self.trainer.save_all()

    def train(self):
        self.trainer.train()