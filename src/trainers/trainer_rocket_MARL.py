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
                 info: str = "",
                 tqdm_bool: bool = False,
                 print_bool: bool = False):
        super(TrainerEndo, self).__init__(env, worker_agent, central_agent, num_episodes, save_interval, number_of_agents, info, tqdm_bool, print_bool)

    def test_env(self):
        test_agent_interaction(self.env, self.central_agent, self.env.dt, self.print_bool)

class VerticalRisingTrain:
    def __init__(self,
                 num_episodes : int,
                 worker_agent_config : dict,
                 central_agent_config : dict,
                 print_bool : bool = False,
                 tqdm_bool : bool = True,
                 save_interval : int = 10,
                 number_of_agents : int = 2,
                 info : str = "",
                 marl_load_info : str = None):
        self.env = env(sizing_needed_bool = False,
                       print_bool = print_bool)
        if marl_load_info is not None:
            self.load_agents(marl_load_info)
        else:
            worker_agent_config['save_path'] =  'results/VerticalRising-MARL/'
            worker_agent_config['print_bool'] = print_bool

            worker_agent_clone = Agent(
                seed = 0,
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                **worker_agent_config)
            
            central_agent_config['save_path'] =  'results/VerticalRising-MARL/'
            central_agent_config['print_bool'] = print_bool
            
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
                                            info,
                                            tqdm_bool,
                                            print_bool)
            
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.number_of_agents = number_of_agents
        self.info = info
        self.tqdm_bool = tqdm_bool
        self.print_bool = print_bool
    
    def load_agents(self, info : str):
        # Load central agent
        central_agent_path = f'data/agent_saves/VerticalRising-MARL/soft-actor-critic_{info}.pkl'
        central_agent = load_sac(central_agent_path)

        # Load worker agents
        worker_agents = []
        base_path = f'data/agent_saves/VerticalRising-MARL/'
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
                                       self.info,
                                       self.tqdm_bool,
                                       self.print_bool)
        self.trainer.load_all(central_agent, worker_agents)

    def save_all(self):
        self.trainer.save_all()

    def train(self):
        self.trainer.train()