import numpy as np
import math
import matplotlib.pyplot as plt

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
        self.states = []

    def update(self, u0): # i.e. step
        # Actions: u0, u1, for max throttle u1 = 0
        actions = np.array([u0, 0])
        self.state, _ = self.physics_step_func(self.state, actions)
        self.states.append(self.state)
        # Theta is the flight path angle
        theta = self.state[4]
        return theta
    
    def plot_physics_state(self, gamma_reference_array, gimbal_angles_deg):
        # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
        xs = [state[0] for state in self.states]
        ys = [state[1]/1000 for state in self.states]
        vxs = [state[2] for state in self.states]
        vys = [state[3]/1000 for state in self.states]
        thetas = [math.degrees(state[4]) for state in self.states]
        theta_dots = [math.degrees(state[5]) for state in self.states]
        gammas = [math.degrees(state[6]) for state in self.states]
        alphas = [math.degrees(state[7]) for state in self.states]
        masses = [state[8] for state in self.states]
        mass_propellants = [state[9] for state in self.states]
        # Plot 1) time vs x 2) time vs y 3) time vs vx 4) time vs vy 5) time vs theta 6) time vs theta_dot 7) time vs gamma 8) time vs alpha 9) time vs mass 10) time vs mass_propellant
        time_array = np.arange(0, self.delta_t*len(self.states), self.delta_t)
        fig, ax = plt.subplots(4, 2, figsize=(12, 12))
        ax[0, 0].plot(time_array, xs)
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].set_ylabel('x (m)')
        ax[0, 0].set_title('x')
        ax[0, 0].grid()

        ax[0, 1].plot(time_array, ys)
        ax[0, 1].set_xlabel('Time (s)')
        ax[0, 1].set_ylabel('y (km)')
        ax[0, 1].set_title('y')
        ax[0, 1].grid()

        ax[1, 0].plot(time_array, vxs)
        ax[1, 0].set_xlabel('Time (s)')
        ax[1, 0].set_ylabel('vx (m/s)')
        ax[1, 0].set_title('vx')
        ax[1, 0].grid()

        ax[1, 1].plot(time_array, vys)
        ax[1, 1].set_xlabel('Time (s)')
        ax[1, 1].set_ylabel('vy (km/s)')
        ax[1, 1].set_title('vy')
        ax[1, 1].grid()

        ax[2, 0].plot(time_array, thetas, label = 'Pitch Angle  ')
        ax[2, 0].plot(time_array, gammas, label = 'Flight Path Angle')
        ax[2, 0].plot(time_array, gamma_reference_array, label = 'Flight Path Angle Reference')
        ax[2, 0].set_xlabel('Time (s)')
        ax[2, 0].set_ylabel('Angle (deg)')
        ax[2, 0].set_title('Flight Path Angle and Pitch Angle')
        ax[2, 0].grid()
        ax[2, 0].legend()

        ax[2, 1].plot(time_array, alphas)
        ax[2, 1].set_xlabel('Time (s)')
        ax[2, 1].set_ylabel('alpha (deg)')
        ax[2, 1].set_title('alpha')
        ax[2, 1].grid()

        ax[3, 0].plot(time_array, gimbal_angles_deg)
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 0].set_ylabel('Gimbal Angle (deg)')
        ax[3, 0].set_title('Gimbal Angle')
        ax[3, 0].grid()
        
        
        plt.tight_layout()
        plt.show()

    def reset(self):
        self.state = self.initial_state
        self.states = []
# Generate reference array
delta_t = get_dt()
reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
reference_state_func = lambda t : generate_reference_state(reference_trajectory_func, t)
t = 0
gamma_reference_array = []
while t < final_reference_time:
    gamma_reference_array.append(reference_state_func(t))
    t += delta_t
# Fix initial gamma to next gamma reference
gamma_reference_array[0] = gamma_reference_array[1]


# Closed loop model
max_abs_gimbal_angle = math.radians(30)
max_gimbal_rate = math.radians(5)
# Controller is : gimbal_angle_rad = math.radians(30)* np.clip(2*u0, -1, 1)
controller_RL = max_gimbal_rate / max_abs_gimbal_angle
params = {
    'kp': 0.9,
    'ki': 0,
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


class AttitudeCloseLoopFB(ClosedLoopFB):
    def __init__(self, params, attitude_model, delta_t):
        super().__init__(params, attitude_model, delta_t)

    def plot_results(self):
        gimbal_angles_rad = math.radians(30)* np.clip(2*np.array(self.controller_output_array), -1, 1)
        gimbal_angles_deg = np.rad2deg(gimbal_angles_rad)
        reference_flight_path_angles_deg = np.rad2deg(np.array(self.reference_array))
        attitude_model.plot_physics_state(reference_flight_path_angles_deg, gimbal_angles_deg)
        '''
        time_array = np.arange(0, self.delta_t*len(self.reference_array), self.delta_t)
        # Angle of attack = pitch angle - flight path angle i.e. output - reference
        flight_path_angles_deg = np.rad2deg(np.array(self.reference_array))
        pitch_angles_deg = np.rad2deg(np.array(self.output_array))
        angles_of_attack = pitch_angles_deg - flight_path_angles_deg

        gimbal_angles_deg = np.rad2deg(np.array(self.controller_output_array))
        # 3 plots: 1) Reference tracking in degrees 2) Gimbal angle in degrees 3) Angle of Attack in degrees
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))

        ax[0].plot(time_array, flight_path_angles_deg, label='Flight Path Angle (Reference)')
        ax[0].plot(time_array, pitch_angles_deg, label='Pitch Angle (Output)')
        ax[0].legend()
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Output (deg)')
        ax[0].set_title('Flight Path Angle and Pitch Angle')

        ax[1].plot(time_array, gimbal_angles_deg, label='Gimbal Angle')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Gimbal Angle (deg)')  
        ax[1].set_title('Gimbal Angle')

        ax[2].plot(time_array, angles_of_attack, label='Angle of Attack')
        ax[2].legend()
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Angle of Attack (deg)')
        ax[2].set_title('Angle of Attack')
        plt.tight_layout()
        plt.show()
        '''
    

closed_loop_fb = AttitudeCloseLoopFB(params, attitude_model, delta_t)

# Run closed loop model
results = closed_loop_fb.update_array(gamma_reference_array)
# Plot results
closed_loop_fb.plot_results()