from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.rockets_physics import compile_physics
from src.envs.rl.rtd_rl import compile_rtd_rl
from src.envs.pso.rtd_pso import compile_rtd_pso
from src.envs.utils.reference_trajectory_interpolation import get_dt
from src.RocketSizing.main_sizing import size_rocket

class rocket_environment_pre_wrap:
    def __init__(self,
                 sizing_needed_bool = False,
                 type = 'rl',
                 flight_stage = 'subsonic'):
        # Ensure state_initial is set before run_test_physics
        self.dt = get_dt()
        self.physics_step, self.state_initial = compile_physics(self.dt)
        self.state = self.state_initial
        self.type = type

        if sizing_needed_bool:
            size_rocket()
            self.run_test_physics()

        assert type in ['rl', 'pso']

        if type == 'rl':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_rl()
        elif type == 'pso':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_pso(flight_stage = flight_stage)

        # Startup sequence
        self.reset()
        if type == 'pso':
            self.truncation_id = 0

    def reset(self):
        self.state = self.state_initial
        if self.type == 'pso':
            self.truncation_id = 0
        return self.state

    def step(self, actions):
        # Physics step
        self.state, info = self.physics_step(self.state,
                                             actions)
        info['state'] = self.state
        info['actions'] = actions

        if self.type == 'pso':
            truncated, self.truncation_id = self.truncated_func(self.state)
        else:
            truncated = self.truncated_func(self.state)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated)        

        return self.state, reward, done, truncated, info
    
    def run_test_physics(self):
        universal_physics_plotter(env = self,
                                  agent = None,
                                  save_path = f'results/physics_test/',
                                  type = 'physics')
        self.reset()

    def physics_step_test(self, actions, target_altitude):
        terminated = False
        self.state, info = self.physics_step(self.state, actions)
        info['state'] = self.state
        info['actions'] = actions
        altitude = self.state[1]
        propellant_mass = self.state[-2]
        if altitude >= target_altitude:
            terminated = True
        elif propellant_mass <= 0:
            terminated = True
        else:
            terminated = False

        return self.state, terminated, info
    

