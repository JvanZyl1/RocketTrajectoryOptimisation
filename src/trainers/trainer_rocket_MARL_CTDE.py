import os

from src.envs.env_ascent import ascent_wrapped_env as env
from src.envs.env_endo.physics_plotter import test_agent_interaction
from src.agents.functions.load_agent import load_sac_marl_ctde
from src.agents.sac_marl_ctde import SAC_MARL_CTDE

from src.trainers.trainers import Trainer_MARL_CTDE

class TrainerEndo(Trainer_MARL_CTDE):
    def __init__(self,
                 env,
                 marl_ctde_agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 info: str = "",
                 tqdm_bool: bool = False,
                 print_bool: bool = False):
        super(TrainerEndo, self).__init__(env, marl_ctde_agent, num_episodes, save_interval, info, tqdm_bool, print_bool)

    def test_env(self):
        test_agent_interaction(self.env,
                               self.marl_ctde_agent,
                               self.env.dt,
                               self.print_bool)
        


class VerticalRisingTrain:
    def __init__(self,
                 number_of_episodes : int,
                 agent_config : dict,
                 info : str = "",
                 save_interval : int = 20,
                 tqdm_bool : bool = True,
                 print_bool : bool = False):

        self.number_of_episodes = number_of_episodes

        self.env = env(sizing_needed_bool = False,
                       print_bool = print_bool)
        
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        seed = 0

        save_path_rewards = 'results/VerticalRising-SAC-MARL-CTDE/'

        self.agent = SAC_MARL_CTDE(seed = seed,
                                   state_dim = state_dim,
                                   action_dim = action_dim,
                                   config =agent_config,
                                   save_path = save_path_rewards,
                                   print_bool = print_bool)
        
        self.trainer = TrainerEndo(env = self.env,
                                                      marl_ctde_agent = self.agent,
                                                      num_episodes = self.number_of_episodes,
                                                      save_interval = save_interval,
                                                      info = info,
                                                      tqdm_bool = tqdm_bool,
                                                      print_bool = print_bool)
        
        self.tqdm_bool = tqdm_bool
        self.print_bool = print_bool
        self.save_interval = save_interval
        self.info = info

    def load_agent(self, info : str):
        agent_path = f'data/agents_saves/VerticalRising-SAC-MARL-CTDE/soft-actor-critic-marl-ctde_{info}.pkl'
        self.agent = load_sac_marl_ctde(agent_path)
        self.trainer = Trainer_MARL_CTDE(env = self.env,
                                      marl_ctde_agent = self.agent,
                                      num_episodes = self.number_of_episodes,
                                      save_interval = self.save_interval,
                                      info = self.info,
                                      tqdm_bool = self.tqdm_bool,
                                      print_bool = self.print_bool)
        
    def train(self):
        self.trainer.train()

    def save_all(self):
        self.trainer.save_all()