import numpy as np
import math

from src.envs.env_endo.init_vertical_rising import get_dt
from src.envs.env_endo.init_vertical_rising import reference_trajectory_lambda
from src.envs.env_endo.physics_endo import setup_physics_step_endo

from src.controls.RotationController.controller_utils import ClosedLoopFB

def generate_reference_state(reference_func, time):
    reference_state = reference_func(time)
    xr, yr, vxr, vyr, m = reference_state

    # Flight path angle
    gammar = np.arctan2(vyr, vxr)

    # Theta reference = Flight path angle reference; so alpha = 0
    return gammar

class AttitudeModel:
    def __init__(self, delta_t):
        self.physics_step_func, self.state = setup_physics_step_endo(delta_t)
        self.initial_state = self.state
        self.delta_t = delta_t

    def update(self, gimbal_angle): # i.e. step
        # Actions: u0, u1, for max throttle u0 = 0
        actions = np.array([gimbal_angle, 0])
        self.state, _ = self.physics_step_func(self.state, actions)
        # Theta is the flight path angle
        return gamma

    def reset(self):
        self.state = self.initial_state

# Generate reference array
delta_t = get_dt()
reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
reference_state_func = lambda t : generate_reference_state(reference_trajectory_func, t)
t = 0
gamma_reference_array = []
while t < final_reference_time:
    gamma_reference_array.append(reference_state_func(t))
    t += delta_t


# Closed loop model
max_abs_gimbal_angle = math.radians(30)
max_gimbal_rate = math.radians(5)
# Controller is : gimbal_angle_rad = math.radians(30)* np.clip(2*u0, -1, 1)
controller_RL = max_gimbal_rate / max_abs_gimbal_angle
params = {
    'kp': -0.1,
    'ki': -0.0005,
    'kd': 0,
    'reference_RL': math.radians(1),                    # Max reference rate is 1 rad/s
    'reference_saturation_lower': -math.radians(90),    # Maximum flight path angle error is 10 deg
    'reference_saturation_upper': math.radians(90),     #    
    'controller_RL': controller_RL,                      # Maximum gimbal rate is 5 deg/s
    'controller_saturation_lower': -1,   # Controller normalisation
    'controller_saturation_upper': 1,
    'initial_reference_t': gamma_reference_array[0],
}

attitude_model = AttitudeModel(delta_t)
closed_loop_fb = ClosedLoopFB(params, attitude_model, delta_t)

# Run closed loop model
results = closed_loop_fb.update_array(gamma_reference_array)

# Plot results
closed_loop_fb.plot_results()