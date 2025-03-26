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
                    info: str = ""):
            super(TrainerEndo, self).__init__(env, agent, num_episodes, save_interval, info)

    def test_env(self):
        test_agent_interaction(self.env,
                                 self.agent)

class VerticalRisingTrain:
    def __init__(self,
                 agent_config : dict,
                 number_of_episodes : int = 250,
                 save_interval : int = 10,
                 info : str = "",
                 debug_mode : bool = False,
                 actor_params : dict = None): # To load the parameters from the particle swarm optimisation
        self.num_episodes = number_of_episodes
        seed = 0

        self.env = env(sizing_needed_bool = False)
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        self.model_name = 'VerticalRising-SAC'

        agent_config['model_name'] = self.model_name
        agent_config['print_bool'] = debug_mode                          # Set to True to debug due to jax

        self.agent = Agent(
            seed = seed,
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config)
        if actor_params is not None:
            self.agent.actor_params = actor_params

        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = save_interval,
                               info = info)
        
        self.save_interval = save_interval

    def load_agent(self, info : str):
        self.agent = load_sac(f'data/agent_saves/{self.model_name}/saves/soft-actor-critic_{info}.pkl')
        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = self.save_interval,
                               info = info)     

    def train(self):
        self.trainer.train()

    def save_all(self):
        self.trainer.save_all()

