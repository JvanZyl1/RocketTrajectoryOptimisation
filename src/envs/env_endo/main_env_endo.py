from src.envs.env_endo.physics_plotter import test_physics_endo_with_plot as test_physics
from src.envs.env_endo.physics_endo import setup_physics_step_endo as setup_physics
from src.envs.env_endo.init_vertical_rising import create_quick_reward_func

class rocket_model_endo_ascent:
    def __init__(self,
                 dt : float):

        self.dt = dt
        self.truncation_id = 0
        self.startup()
        self.throttle_allowed_bool = False # Vertical rising, True for gravity turn

        self.reward_func = create_quick_reward_func()


    def reset(self):
        self.physics_state = self.physics_state_initial

    def startup(self):
        self.physics_step, self.physics_state_initial = setup_physics(self.dt)
        self.physics_state = self.physics_state_initial
        test_physics(self)
        self.reset()

    def augment_state(self):
        # Augment observation to vx and vy only
        return self.physics_state[2:4]

    def step(self, actions):
        # Physics step
        self.physics_state, info = self.physics_step(self.physics_state,
                                                     actions,
                                                     self.throttle_allowed_bool)
        info['physics_state'] = self.physics_state

        # Augment state
        self.agent_state = self.augment_state()
        # Truncated function
        truncated = self.truncated_func()
        # Done function
        done = self.done_func()
        # Reward function
        reward = self.reward_func(actions, done, truncated)        

        return self.agent_state, reward, done, truncated, info