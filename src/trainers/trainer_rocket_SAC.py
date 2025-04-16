import jax
import jax.numpy as jnp

from src.trainers.trainers import TrainerSAC
from src.envs.universal_physics_plotter import universal_physics_plotter

from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.functions.load_agent import load_sac
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env

class TrainerEndo(TrainerSAC):
    def __init__(self,
                    env,
                    agent,
                    num_episodes: int,
                    save_interval: int = 10,
                    info: str = "",
                    critic_warm_up_steps: int = 0,
                    experiences_model_name: str = None):
            super(TrainerEndo, self).__init__(env, agent, num_episodes, save_interval, info, critic_warm_up_steps, experiences_model_name)

    def test_env(self):
        universal_physics_plotter(self.env,
                                  self.agent,
                                  self.agent.save_path,
                                  type = 'rl')
class RocketTrainer_SAC:
    def __init__(self,
                 agent_config : dict,
                 number_of_episodes : int = 250,
                 save_interval : int = 10,
                 info : str = "",
                 actor_params : dict = None,
                 critic_params : jnp.ndarray = None,
                 critic_target_params : jnp.ndarray = None,
                 critic_opt_state : jnp.ndarray = None,
                 critic_warm_up_steps : int = 0,
                 experiences_model_name : str = None,
                 flight_phase : str = 'subsonic'): # To load the parameters from the particle swarm optimisation
        
        self.experiences_model_name = experiences_model_name
        self.num_episodes = number_of_episodes

        self.env = env(sizing_needed_bool = False,
                       flight_phase = flight_phase)
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        self.model_name = 'VerticalRising-SAC'

        agent_config['model_name'] = self.model_name
        self.agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            **agent_config)

        if actor_params is not None:
            self.agent.actor_params = actor_params

        if critic_params is not None:
            self.agent.critic_params = critic_params

        if critic_target_params is not None:
            self.agent.critic_target_params = critic_target_params

        if critic_opt_state is not None:
            self.agent.critic_opt_state = critic_opt_state

        self.critic_warm_up_steps = critic_warm_up_steps

        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = save_interval,
                               info = info,
                               critic_warm_up_steps = self.critic_warm_up_steps,
                               experiences_model_name = self.experiences_model_name)
        
        self.save_interval = save_interval

    def load_agent(self, info : str):
        self.agent = load_sac(f'data/agent_saves/{self.model_name}/saves/soft-actor-critic_{info}.pkl')
        self.trainer = TrainerEndo(env   = self.env,
                               agent = self.agent,
                               num_episodes = self.num_episodes,
                               save_interval = self.save_interval,
                               info = info,
                               critic_warm_up_steps = self.critic_warm_up_steps,
                               experiences_model_name = self.experiences_model_name)

    def train(self):
        self.trainer.train()

    def save_all(self):
        self.trainer.save_all()

