import os

from src.trainers.trainers import TrainerSAC
from src.envs.env_endo.physics_plotter import test_agent_interaction

from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.functions.load_agent import load_sac
from src.envs.env_ascent import ascent_wrapped_env as env

class TrainerEndo(TrainerSAC):
    def __init__(self,
                    env,
                    agent,
                    num_episodes: int,
                    save_interval: int = 10,
                    info: str = "",
                    tqdm_bool: bool = False,
                    print_bool: bool = False):
            super(TrainerEndo, self).__init__(env, agent, num_episodes, save_interval, info, tqdm_bool, print_bool)

    def test_env(self):
        test_agent_interaction(self.env,
                                 self.agent,
                                 self.env.dt,
                                 self.print_bool)

class VerticalRisingTrain:
    def __init__(self,
                 agent_config : dict,
                 number_of_episodes : int = 250,
                 save_interval : int = 10,
                 info : str = "",
                 actor_params : dict = None, # To load the parameters from the particle swarm optimisation
                 tqdm_bool : bool = True,
                 print_bool : bool= False):
        self.num_episodes = number_of_episodes
        seed = 0

        self.env = env(sizing_needed_bool = False,
                       print_bool = print_bool)
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        save_path_rewards = 'results/VerticalRising-SAC/'

        agent_config['save_path'] = save_path_rewards
        agent_config['print_bool'] = print_bool

        self.agent = Agent(
            seed = seed,
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config)
        self.agent.actor_params = actor_params
        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = save_interval,
                               info = info,
                               tqdm_bool = tqdm_bool,
                               print_bool = print_bool)
        
        self.print_bool = print_bool
        self.tqdm_bool = tqdm_bool
        self.save_interval = save_interval

    def load_agent(self, info : str):
        agent_path = os.path.join('..', 'data', 'agents_saves', 'VerticalRising-SAC', f'soft-actor-critic_{info}.pkl')
        self.agent = load_sac(agent_path)
        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = self.save_interval,
                               info = info,
                               tqdm_bool = self.tqdm_bool,
                               print_bool = self.print_bool)     

    def train(self):
        self.trainer.train()

    def save_all(self):
        self.trainer.save_all()

