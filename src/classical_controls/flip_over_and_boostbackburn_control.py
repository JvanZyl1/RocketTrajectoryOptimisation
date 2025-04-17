import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.envs.base_environment import load_flip_over_initial_state
from src.envs.rockets_physics import compile_physics
def PD_controller_single_step(Kp, Kd, N, error, previous_error, previous_derivative, dt):
    # Proportional term
    P_term = Kp * error
    
    # Derivative term with low-pass filter
    derivative = (error - previous_error) / dt
    D_term = Kd * (N * derivative + (1 - N * dt) * previous_derivative)
    
    # Control action
    control_action = P_term + D_term
    
    return control_action, derivative

def flip_over_pitch_control(pitch_angle_rad, max_gimbal_angle_deg, previous_pitch_angle_error_rad, previous_derivative, dt, flip_over_pitch_reference_deg):
    Kp_theta_flip = -40
    Kd_theta_flip = -20
    N_theta_flip = 14

    pitch_angle_error_rad = math.radians(flip_over_pitch_reference_deg) - pitch_angle_rad
    gimbal_angle_command_deg, new_derivative = PD_controller_single_step(Kp=Kp_theta_flip,
                                                                         Kd=Kd_theta_flip,
                                                                         N=N_theta_flip,
                                                                         error=pitch_angle_error_rad,
                                                                         previous_error=previous_pitch_angle_error_rad,
                                                                         previous_derivative=previous_derivative,
                                                                         dt=dt)

    # Clip the result
    gimbal_angle_command_deg = np.clip(gimbal_angle_command_deg, -max_gimbal_angle_deg, max_gimbal_angle_deg)

    return gimbal_angle_command_deg, pitch_angle_error_rad, new_derivative

def augment_action_flip_over(action,
                             max_gimbal_angle_deg):
    return action/max_gimbal_angle_deg

class FlipOverandBoostbackBurnControl:
    def __init__(self,
                 pitch_tuning_bool : bool = False):
        self.dt = 0.1
        self.max_gimbal_angle_deg = 45
        self.final_pitch_error_deg = 2
        self.flip_over_pitch_reference_deg = 175
        self.vx_terminal = -150
        self.pitch_tuning_bool = pitch_tuning_bool

        self.pitch_controller_lambda = lambda pitch_angle_rad, previous_pitch_angle_error_rad, previous_derivative : flip_over_pitch_control(pitch_angle_rad = pitch_angle_rad,
                                                                                                                                    max_gimbal_angle_deg = self.max_gimbal_angle_deg,
                                                                                                                                    previous_pitch_angle_error_rad = previous_pitch_angle_error_rad,
                                                                                                                                    previous_derivative = previous_derivative,
                                                                                                                                    dt = self.dt,
                                                                                                                                    flip_over_pitch_reference_deg = self.flip_over_pitch_reference_deg)
        
        self.state = load_flip_over_initial_state()
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 'flip_over_boostbackburn')
        self.initialise_logging()
        self.initial_conditions()
        
    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.pitch_angle_deg_vals = []
        self.pitch_angle_reference_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.angle_of_attack_deg_vals = []
        self.gimbal_angle_commanded_deg_vals = []
        self.u0_vals = []

        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.gimbal_angle_deg_vals = []
    def initial_conditions(self):
        self.gimbal_angle = 0.0
        self.previous_pitch_angle_error_rad = 0.0
        self.pitch_angle_previous_derivative = 0.0

    def reset(self):
        # Reset state and previous values
        self.state, _ = load_flip_over_initial_state()
        self.previous_pitch_angle_error_rad = 0.0
        self.pitch_angle_previous_derivative = 0.0
        self.initialise_logging()

    def closed_loop_step(self):
        gimbal_angle_command_deg, self.previous_pitch_angle_error_rad, self.pitch_angle_previous_derivative = self.pitch_controller_lambda(pitch_angle_rad = self.state[4],
                                                                                                      previous_pitch_angle_error_rad = self.previous_pitch_angle_error_rad,
                                                                                                      previous_derivative = self.pitch_angle_previous_derivative)
        action = augment_action_flip_over(action = gimbal_angle_command_deg,
                                          max_gimbal_angle_deg = self.max_gimbal_angle_deg)
        
        self.state, info = self.simulation_step_lambda(self.state, action, self.gimbal_angle)
        self.gimbal_angle = info['action_info']['gimbal_angle_deg']
        # state : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.state[4]))
        self.pitch_angle_reference_deg_vals.append(math.degrees(math.radians(175)))
        self.time_vals.append(self.state[-1])
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(info['mach_number'])
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.gimbal_angle_commanded_deg_vals.append(gimbal_angle_command_deg)
        self.gimbal_angle_deg_vals.append(self.gimbal_angle)
        self.u0_vals.append(action)
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.mass_propellant_vals.append(self.state[9])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])

    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/flip_over_and_boostbackburn_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_flip_over_and_boostbackburn_control.csv')

        # Create a DataFrame from the collected data
        data = {
            't[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'mass[kg]': self.mass_vals
        }
        
        # Save the DataFrame to a CSV file
        pd.DataFrame(data).to_csv(full_trajectory_path, index=False)

        pitch_angle_rad_vals = np.array(self.pitch_angle_deg_vals) * np.pi / 180
        pitch_rate_rad_vals = np.array(self.pitch_rate_deg_vals) * np.pi / 180
        angle_of_attack_rad_vals = np.array(self.angle_of_attack_deg_vals) * np.pi / 180
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': pitch_angle_rad_vals,
            'theta_dot[rad/s]': pitch_rate_rad_vals,
            'alpha[rad]': angle_of_attack_rad_vals,
            'mass[kg]': self.mass_vals,
            'gimbalanglecommanded[deg]': self.gimbal_angle_commanded_deg_vals,
            'u0': self.u0_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_flip_over_and_boostbackburn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        # A4 size plot
        plt.figure(figsize=(8.27, 11.69))
        plt.subplot(4, 2, 1)
        plt.plot(self.x_vals, self.y_vals, linewidth = 2)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Flight Path')
        plt.grid()

        plt.subplot(4, 2, 2)
        plt.plot(self.time_vals, self.mach_number_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Mach Number')
        plt.title('Mach Number')
        plt.grid()

        plt.subplot(4, 2, 3)
        plt.plot(self.time_vals, self.vx_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity x')
        plt.grid()

        plt.subplot(4, 2, 4)
        plt.plot(self.time_vals, self.vy_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity y')
        plt.grid()

        plt.subplot(4, 2, 5)
        plt.plot(self.time_vals, self.pitch_angle_deg_vals, linewidth = 2, label = 'Pitch Angle')
        plt.plot(self.time_vals, self.pitch_angle_reference_deg_vals, linewidth = 2, label = 'Pitch Angle Reference')
        plt.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 2, label = 'Flight Path Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Pitch and Flight Path Angles')
        plt.grid()
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(self.time_vals, self.angle_of_attack_deg_vals, linewidth = 2, label = 'Angle of Attack')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Angle of Attack')
        plt.grid()

        plt.subplot(4, 2, 7)
        plt.plot(self.time_vals, self.gimbal_angle_commanded_deg_vals, linewidth = 2, label = 'Commanded')
        plt.plot(self.time_vals, self.gimbal_angle_deg_vals, linewidth = 2, label = 'Actual')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Gimbal Angle (Commanded and Actual)')
        plt.grid()
        plt.legend()
        plt.subplot(4, 2, 8)
        plt.plot(self.time_vals, self.mass_propellant_vals, linewidth = 2, label = 'Mass Propellant')
        plt.plot(self.time_vals, self.mass_vals, linewidth = 2, label = 'Mass')
        plt.xlabel('Time [s]')
        plt.ylabel('Mass [kg]')
        plt.title('Mass Propellant and Mass')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if self.pitch_tuning_bool:
            plt.savefig(f'results/classical_controllers/flip_over_and_boostbackburn_pitch_tuning.png')
        else:
            plt.savefig(f'results/classical_controllers/flip_over_and_boostbackburn.png')
        plt.close()

    def run_closed_loop(self):
        # vx < - 150 m/s
        if self.pitch_tuning_bool:
            pitch_angle_error_deg = self.flip_over_pitch_reference_deg - math.degrees(self.state[4])
            time_settled = 0.0
            time_ran = 0.0
            while time_settled < 1 and time_ran < 200:
                self.closed_loop_step()
                pitch_angle_error_deg = abs(self.flip_over_pitch_reference_deg - math.degrees(self.state[4]))
                time_ran += self.dt
                if pitch_angle_error_deg < self.final_pitch_error_deg:
                    time_settled += self.dt
                else:
                    time_settled = 0.0
        else:
            time_ran = 0.0
            vx = self.state[2]
            while vx > self.vx_terminal and time_ran < 80:
                self.closed_loop_step()
                vx = self.state[2]
                time_ran += self.dt
        self.plot_results()
        self.save_results()