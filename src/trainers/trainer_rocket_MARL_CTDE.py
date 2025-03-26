from src.envs.env_ascent import ascent_wrapped_env as env
from src.envs.env_endo.physics_plotter import test_agent_interaction_reinforcement_learning
from src.agents.functions.load_agent import load_sac_marl_ctde
from src.agents.sac_marl_ctde import SAC_MARL_CTDE

from src.trainers.trainers import Trainer_MARL_CTDE

class TrainerEndo(Trainer_MARL_CTDE):
    def __init__(self,
                 env,
                 marl_ctde_agent,
                 num_episodes: int,
                 save_interval: int = 10,
                 info: str = ""):
        super(TrainerEndo, self).__init__(env, marl_ctde_agent, num_episodes, save_interval, info)

    def test_env(self):
        test_agent_interaction_reinforcement_learning(self.env,
                               self.marl_ctde_agent)
        


class VerticalRisingTrain:
    def __init__(self,
                 number_of_episodes : int,
                 agent_config : dict,
                 info : str = "",
                 save_interval : int = 20,
                 debug_mode : bool = False):

        self.number_of_episodes = number_of_episodes

        self.env = env(sizing_needed_bool = False)
        
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        seed = 0

        self.model_name = 'VerticalRising-SAC-MARL-CTDE'

        self.agent = SAC_MARL_CTDE(seed = seed,
                                   state_dim = state_dim,
                                   action_dim = action_dim,
                                   config =agent_config,
                                   model_name = self.model_name,
                                   print_bool = debug_mode)
        
        self.trainer = TrainerEndo(env = self.env,
                                                      marl_ctde_agent = self.agent,
                                                      num_episodes = self.number_of_episodes,
                                                      save_interval = save_interval,
                                                      info = info)

        self.save_interval = save_interval
        self.info = info

    def load_agent(self, info : str):
        self.agent = load_sac_marl_ctde(f'data/agent_saves/{self.model_name}/saves/soft-actor-critic-marl-ctde_{info}.pkl')
        self.trainer = Trainer_MARL_CTDE(env = self.env,
                                      marl_ctde_agent = self.agent,
                                      num_episodes = self.number_of_episodes,
                                      save_interval = self.save_interval,
                                      info = self.info)
        
    def train(self):
        self.trainer.train()

    def save_all(self):
        self.trainer.save_all()