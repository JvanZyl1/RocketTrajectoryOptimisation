import jax.numpy as jnp

from src.trainers.trainers import TrainerSAC
from src.envs.universal_physics_plotter import universal_physics_plotter
from configs.agent_config import config_subsonic, config_supersonic, config_flip_over_boostbackburn, config_ballistic_arc_descent
from src.agents.soft_actor_critic import SoftActorCritic as Agent
from src.agents.functions.load_agent import load_sac
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.critic_pre_train.pre_train_critic import pre_train_critic_from_pso_experiences
from src.envs.supervisory.agent_load_supervisory import load_supervisory_actor

class TrainerEndo(TrainerSAC):
    def __init__(self,
                 env,
                 agent,
                 flight_phase : str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 load_buffer_from_experiences_bool : bool = False):
        super(TrainerEndo, self).__init__(env, agent, flight_phase, num_episodes, save_interval, critic_warm_up_steps, load_buffer_from_experiences_bool)

    def test_env(self):
        universal_physics_plotter(self.env,
                                  self.agent,
                                  self.agent.save_path,
                                  flight_phase = self.env.flight_phase,
                                  type = 'rl')
class RocketTrainer_SAC:
    def __init__(self,
                 flight_phase : str,
                 save_interval : int = 10,
                 load_from : str = 'None',
                 load_buffer_bool : bool = False,
                 pre_train_critic_bool : bool = False):
        self.flight_phase = flight_phase
        self.env = env(flight_phase = flight_phase)

        if flight_phase == 'subsonic':
            self.agent_config = config_subsonic
        elif flight_phase == 'supersonic':
            self.agent_config = config_supersonic
        elif flight_phase == 'flip_over_boostbackburn':
            self.agent_config = config_flip_over_boostbackburn
        elif flight_phase == 'ballistic_arc_descent':
            self.agent_config = config_ballistic_arc_descent

        if load_from == 'pso':
            self.load_agent_from_pso()
        elif load_from == 'rl':
            self.load_agent_from_rl()
        elif load_from == 'supervisory':
            self.load_agent_from_supervisory()
        else:
            self.agent = Agent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase = self.flight_phase,
                **self.agent_config['sac'])
            
        if pre_train_critic_bool:
            self.pre_train_critic()

        self.trainer = TrainerEndo(env   = self.env,
                                   agent = self.agent,
                                   flight_phase = self.flight_phase,
                                   num_episodes = self.agent_config['num_episodes'],
                                   save_interval = save_interval,
                                   critic_warm_up_steps = self.agent_config['critic_warm_up_steps'],
                                   load_buffer_from_experiences_bool = load_buffer_bool)
        
        self.save_interval = save_interval

    def __call__(self):
        self.trainer.train()

    def load_agent_from_pso(self):
        actor_params, hidden_dim, number_of_hidden_layers = load_pso_actor(self.flight_phase)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = number_of_hidden_layers

        self.agent = Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            flight_phase = self.flight_phase,
            **self.agent_config['sac'])
        self.agent.actor_params = actor_params  

    def load_agent_from_rl(self):
        self.agent = load_sac(f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/soft-actor-critic.pkl')

    def load_agent_from_supervisory(self):
        actor, actor_params, hidden_dim, hidden_layers = load_supervisory_actor(self.flight_phase)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = hidden_layers
        self.agent = Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            flight_phase = self.flight_phase,
            **self.agent_config['sac'])
        actor_params_clean = {'params' : actor_params}
        self.agent.actor_params = actor_params_clean        
    
    def pre_train_critic(self):
        critic_params_learner = pre_train_critic_from_pso_experiences(flight_phase = self.flight_phase,
                                                                      state_dim = self.env.state_dim,
                                                                      action_dim = self.env.action_dim,
                                                                      hidden_dim_critic = self.agent_config['sac']['hidden_dim_critic'],
                                                                      number_of_hidden_layers_critic = self.agent_config['sac']['number_of_hidden_layers_critic'],
                                                                      gamma = self.agent_config['sac']['gamma'],
                                                                      tau = self.agent_config['sac']['tau'],
                                                                      critic_learning_rate = self.agent_config['pre_train_critic_learning_rate'],
                                                                      batch_size = self.agent_config['pre_train_critic_batch_size'])
        self.agent.critic_params, self.agent.critic_target_params, self.agent.critic_opt_state = critic_params_learner()