from src.envs.env_endo.physics_plotter import test_physics_endo_with_plot as test_physics
from src.envs.env_endo.physics_endo import setup_physics_step_endo as setup_physics
from src.envs.env_endo.init_vertical_rising import create_env_funcs, get_dt
from src.envs.env_setup.main_sizing import size_rocket

class rocket_model_endo_ascent:
    def __init__(self,
                 sizing_needed_bool = False):
        if sizing_needed_bool:
            size_rocket()
            self.run_test_physics()

        self.dt = get_dt()
        self.throttle_allowed_bool = True # Vertical rising, True for gravity turn
        self.reward_func, self.truncated_func, self.done_func = create_env_funcs()

        # Startup sequence
        self.physics_step, self.state_initial = setup_physics(self.dt)
        self.state = self.state_initial
        self.reset()

    def reset(self):
        self.state = self.state_initial
        return self.state

    def step(self, actions):
        # Physics step
        self.state, info = self.physics_step(self.state,
                                             actions)
        info['state'] = self.state
        info['actions'] = actions

        truncated = self.truncated_func(self.state)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated)        

        return self.state, reward, done, truncated, info
    
    def run_test_physics(self):
        test_physics(self)
        self.reset()

    def physics_step_test(self, actions, target_altitude):
        
        terminated = False
        self.state, info = self.physics_step(self.state,
                                                     actions,
                                                     throttle_allowed_bool=False)
        altitude = self.state[1]
        propellant_mass = self.state[-2]
        if altitude >= target_altitude:
            terminated = True
        elif propellant_mass <= 0:
            terminated = True
        else:
            terminated = False

        return self.state, terminated, info
    

