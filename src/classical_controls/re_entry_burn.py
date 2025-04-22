import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.envs.base_environment import load_re_entry_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

def throttle_controller(mach_number, air_density, speed_of_sound, Q_max):
    Kp_mach = 0.15
    Q_ref = Q_max - 5000 # [Pa]
    mach_number_max = math.sqrt(2 * Q_ref / air_density) * 1 / speed_of_sound
    error_mach_number = mach_number_max - mach_number
    throttle = np.clip(Kp_mach * error_mach_number, 0, 1)
    return throttle

def ACS_controller(state, dynamic_pressure,previous_alpha_effective_rad, previous_derivative, max_deflection_angle_deg, dt):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    alpha_effective_rad = gamma - theta - math.pi

    if dynamic_pressure < 5000 and vy < -600:
        Kp_alpha_ballistic_arc = -1
        Kd_alpha_ballistic_arc = 80
        N_alpha_ballistic_arc = 30
    elif dynamic_pressure < 10000 and vy < -600:
        Kp_alpha_ballistic_arc = 50
        Kd_alpha_ballistic_arc = 580.0
        N_alpha_ballistic_arc = 30
    elif dynamic_pressure < 5000 and vy > -600:
        Kp_alpha_ballistic_arc = -6
        Kd_alpha_ballistic_arc = -2
        N_alpha_ballistic_arc = 0
    elif dynamic_pressure < 10000 and vy > -600:
        Kp_alpha_ballistic_arc = 1000
        Kd_alpha_ballistic_arc = 0
        N_alpha_ballistic_arc = 0
    else:
        Kp_alpha_ballistic_arc = -0.308
        Kd_alpha_ballistic_arc = 8.345
        N_alpha_ballistic_arc = 30
        
    delta_norm, new_derivative = PD_controller_single_step(Kp=Kp_alpha_ballistic_arc,
                                                           Kd=Kd_alpha_ballistic_arc,
                                                           N=N_alpha_ballistic_arc,
                                                           error=alpha_effective_rad,
                                                           previous_error=previous_alpha_effective_rad,
                                                           previous_derivative=previous_derivative,
                                                           dt=dt)
    
    if y < 1000:
        delta_norm = 0.0
    delta_norm = np.clip(delta_norm, -1, 1)
    # right up is positive, left down is positive
    delta_left_deg = delta_norm * max_deflection_angle_deg 
    delta_right_deg = delta_left_deg

    return delta_left_deg, delta_right_deg, alpha_effective_rad, new_derivative

def augment_action_ACS(delta_left_deg, delta_right_deg, max_deflection_angle_deg):
    u0 = delta_left_deg / max_deflection_angle_deg
    u1 = delta_right_deg / max_deflection_angle_deg
    return u0, u1

def augment_action_throttle(throttle):
    u2 = 2 * throttle - 1
    return u2

class ReEntryBurn:
    def __init__(self):
        self.dt = 0.1
        self.landing_burn_altitude = 5000
        self.max_deflection_angle_deg = 60
        self.Q_max = 30000 # [Pa]
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 're_entry_burn')
        
        self.acs_controller_lambda = lambda state, dynamic_pressure, previous_alpha_effective_rad, previous_derivative: ACS_controller(state,
                                                                                                                     dynamic_pressure,
                                                                                                                     previous_alpha_effective_rad,
                                                                                                                     previous_derivative,
                                                                                                                     max_deflection_angle_deg = self.max_deflection_angle_deg,
                                                                                                                     dt = self.dt)
        
        self.augment_action_ACS_lambda = lambda delta_left_deg, delta_right_deg: augment_action_ACS(delta_left_deg,
                                                                                                    delta_right_deg,
                                                                                                    max_deflection_angle_deg = self.max_deflection_angle_deg)
        
        self.throttle_controller_lambda = lambda mach_number, air_density, speed_of_sound: throttle_controller(mach_number,
                                                                                                               air_density, 
                                                                                                               speed_of_sound,
                                                                                                               self.Q_max)
        
        self.augment_action_throttle_lambda = lambda throttle: augment_action_throttle(throttle)
        
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
        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.effective_angle_of_attack_deg_vals = []

        self.u0_vals = []
        self.u1_vals = []
        self.u2_vals = []
        self.delta_left_deg_vals = []
        self.delta_right_deg_vals = []

        self.dynamic_pressure_vals = []
        self.control_force_y_vals = []
        self.aero_force_y_vals = []
        self.gravity_force_y_vals = []
    def initial_conditions(self):
        self.delta_left_deg_prev, self.delta_right_deg_prev = 0.0, 0.0
        self.state = load_re_entry_burn_initial_state('supervisory')
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

        self.y0 = self.state[1]
        self.air_density, self.atmospheric_pressure, self.speed_of_sound = endo_atmospheric_model(self.y0)
        speed = math.sqrt(self.state[2]**2 + self.state[3]**2)
        self.mach_number = speed / self.speed_of_sound
        self.dynamic_pressure = 0.5 * self.air_density * speed**2
    def reset(self):
        self.initial_conditions()
        self.initialise_logging()

    def closed_loop_step(self):
        delta_left_deg, delta_right_deg, self.previous_alpha_effective_rad, self.previous_derivative \
            = self.acs_controller_lambda(self.state, self.dynamic_pressure, self.previous_alpha_effective_rad, self.previous_derivative)
        u0, u1 = self.augment_action_ACS_lambda(delta_left_deg, delta_right_deg)
        throttle = self.throttle_controller_lambda(self.mach_number, self.air_density, self.speed_of_sound)
        u2 = self.augment_action_throttle_lambda(throttle)
        actions = (u0, u1, u2)

        self.state, info = self.simulation_step_lambda(self.state, actions, self.delta_left_deg_prev, self.delta_right_deg_prev)
        self.delta_left_deg_prev, self.delta_right_deg_prev = info['action_info']['delta_left_deg'], info['action_info']['delta_right_deg']
        self.air_density, self.speed_of_sound, self.mach_number = info['air_density'], info['speed_of_sound'], info['mach_number']
        self.dynamic_pressure = info['dynamic_pressure']

        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.state[4]))
        self.time_vals.append(self.state[-1])
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(info['mach_number'])
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.mass_propellant_vals.append(self.state[9])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])
        self.effective_angle_of_attack_deg_vals.append(math.degrees(self.state[6] - self.state[4] - math.pi))

        self.u0_vals.append(actions[0])
        self.u1_vals.append(actions[1])
        self.u2_vals.append(actions[2])
        self.delta_left_deg_vals.append(info['action_info']['delta_left_deg'])
        self.delta_right_deg_vals.append(info['action_info']['delta_right_deg'])

        self.dynamic_pressure_vals.append(info['dynamic_pressure'])
        self.control_force_y_vals.append(info['control_force_y'])
        self.aero_force_y_vals.append(info['aero_force_y'])
        self.gravity_force_y_vals.append(info['gravity_force_y'])
    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/re_entry_burn_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_re_entry_burn_control.csv')

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
        flight_path_angle_rad_vals = np.array(self.flight_path_angle_deg_vals) * np.pi / 180
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': pitch_angle_rad_vals,
            'theta_dot[rad/s]': pitch_rate_rad_vals,
            'alpha[rad]': angle_of_attack_rad_vals,
            'gamma[rad]': flight_path_angle_rad_vals,
            'mass[kg]': self.mass_vals,
            'delta_left_deg': self.delta_left_deg_vals,
            'delta_right_deg': self.delta_right_deg_vals,
            'throttle': self.u2_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals,
            'u2': self.u2_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_re_entry_burn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        # A4 size plot
        plt.figure(figsize=(8.27, 11.69))
        plt.subplot(5, 2, 1)
        plt.plot(self.x_vals, self.y_vals, linewidth = 2)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Flight Path')
        plt.grid()

        plt.subplot(5, 2, 2)
        plt.plot(self.time_vals, self.dynamic_pressure_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Dynamic Pressure [Pa]')
        plt.title('Dynamic Pressure')
        plt.grid()

        plt.subplot(5, 2, 3)
        plt.plot(self.time_vals, self.vx_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity x')
        plt.grid()

        plt.subplot(5, 2, 4)
        plt.plot(self.time_vals, self.vy_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity y')
        plt.grid()

        plt.subplot(5, 2, 5)
        plt.plot(self.time_vals, self.pitch_angle_deg_vals, linewidth = 2, label = 'Pitch Angle')
        plt.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 2, label = 'Flight Path Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Pitch Angle')
        plt.grid()
        plt.legend()

        plt.subplot(5, 2, 6)
        plt.plot(self.time_vals, self.effective_angle_of_attack_deg_vals, linewidth = 2, label = 'Effective Angle of Attack')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Effective Angle of Attack')
        plt.grid()

        plt.subplot(5, 2, 7)
        plt.plot(self.time_vals, self.pitch_rate_deg_vals, linewidth = 2, label = 'Pitch Rate')
        plt.xlabel('Time [s]')
        plt.ylabel('Pitch Rate [deg/s]')
        plt.title('Pitch Rate')
        plt.grid()

        plt.subplot(5, 2, 8)
        plt.plot(self.time_vals, self.delta_left_deg_vals, linewidth = 2, label = 'delta_left_deg')
        plt.plot(self.time_vals, self.delta_right_deg_vals, linewidth = 2, linestyle = '--', label = 'delta_right_deg')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('delta_left_deg')
        plt.grid()
        plt.legend()

        plt.subplot(5, 2, 9)
        plt.plot(self.time_vals, self.mass_vals, linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Mass [kg]')
        plt.title('Mass')
        plt.grid()

        plt.subplot(5, 2, 10)
        plt.plot(self.time_vals, self.control_force_y_vals, label = 'Control Force y', linewidth = 2)
        plt.plot(self.time_vals, self.aero_force_y_vals, label = 'Aero Force y', linewidth = 2)
        plt.plot(self.time_vals, self.gravity_force_y_vals, label = 'Gravity Force y', linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel('Force [N]')
        plt.title('Forces - y')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/classical_controllers/re_entry_burn.png')
        plt.close()

    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        while y > self.landing_burn_altitude:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
            if y < 0:
                print(f'y : {y}, vx : {vx}')
                break
        self.plot_results()
        self.save_results()  