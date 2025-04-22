import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.envs.base_environment import load_high_altitude_ballistic_arc_initial_state
from src.envs.rockets_physics import compile_physics
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model


def PD_controller_single_step(Kp, Kd, N, error, previous_error, previous_derivative, dt):
    # Proportional term
    P_term = Kp * error
    
    # Derivative term with low-pass filter
    derivative = (error - previous_error) / dt
    D_term = Kd * (N * derivative + (1 - N * dt) * previous_derivative)
    
    # Control action
    control_action = P_term + D_term
    
    return control_action, derivative

def RCS_force_and_moment_calculator(throttle, # -1 to 1
                                    x_cog,
                                    max_RCS_force_per_thruster,
                                    d_base_rcs_bottom,
                                    d_base_rcs_top):
    thruster_force = max_RCS_force_per_thruster * throttle
    force_bottom = thruster_force * 9
    force_top = thruster_force * 9

    moment_z = -force_bottom * (x_cog - d_base_rcs_bottom) + force_top * (d_base_rcs_top - x_cog)
    return moment_z


def angle_of_attack_controller(state,
                               previous_alpha_effective_rad,
                               previous_derivative,
                               dt):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    alpha_effective_rad = gamma - theta - math.pi

    Kp_alpha_ballistic_arc = 0.8
    Kd_alpha_ballistic_arc = 2.65
    N_alpha_ballistic_arc = 14

    RCS_throttle, new_derivative = PD_controller_single_step(Kp=Kp_alpha_ballistic_arc,
                                                             Kd=Kd_alpha_ballistic_arc,
                                                             N=N_alpha_ballistic_arc,
                                                             error=alpha_effective_rad,
                                                             previous_error=previous_alpha_effective_rad,
                                                             previous_derivative=previous_derivative,
                                                             dt=dt)

    # Clip the result
    RCS_throttle = np.clip(RCS_throttle, -1, 1)
    return RCS_throttle, new_derivative, alpha_effective_rad

class HighAltitudeBallisticArcDescent:
    def __init__(self):
        self.dynamic_pressure_threshold = 2000
        self.dt = 0.1
        with open('data/rocket_parameters/sizing_results.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'max_RCS_force_per_thruster':
                    max_RCS_force_per_thruster = float(row[2])
                if row[0] == 'd_base_rcs_bottom':
                    d_base_rcs_bottom = float(row[2])
                if row[0] == 'd_base_rcs_top':
                    d_base_rcs_top = float(row[2])

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'ballistic_arc_descent')

        self.load_initial_conditions()
        self.initialise_logging()

        self.rcs_force_func_lambda = lambda throttle: RCS_force_and_moment_calculator(throttle, self.x_cog, max_RCS_force_per_thruster, d_base_rcs_bottom, d_base_rcs_top)
        self.rcs_controller_lambda = lambda state, previous_alpha_effective_rad, previous_derivative: angle_of_attack_controller(state, previous_alpha_effective_rad, previous_derivative, self.dt)

    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.pitch_angle_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.angle_of_attack_deg_vals = []
        self.u0_vals = []
        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.effective_angle_of_attack_deg_vals = []

    def load_initial_conditions(self):
        self.state = load_high_altitude_ballistic_arc_initial_state('supervisory')
        _, info_IC = self.simulation_step_lambda(self.state, (0.0))
        self.x_cog = info_IC['x_cog']
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

    def reset(self):
        self.state = load_high_altitude_ballistic_arc_initial_state('supervisory')
        self.initialise_logging()
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

    def closed_loop_step(self):
        RCS_throttle, self.previous_derivative, self.previous_alpha_effective_rad = self.rcs_controller_lambda(self.state,
                                                                                                               self.previous_alpha_effective_rad,
                                                                                                               self.previous_derivative)
        self.state, info = self.simulation_step_lambda(self.state, RCS_throttle)
        effective_angle_of_attack = self.state[6] - self.state[4] - math.pi
        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.state[4]))
        self.time_vals.append(self.state[-1])
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(info['mach_number'])
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.u0_vals.append(RCS_throttle)
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.mass_propellant_vals.append(self.state[9])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])
        self.effective_angle_of_attack_deg_vals.append(math.degrees(self.state[6] - self.state[4] - math.pi))
    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/ballistic_arc_descent_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_ballistic_arc_descent_control.csv')

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
            'u0': self.u0_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_ballistic_arc_descent_control.csv')
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
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Pitch Angle')
        plt.grid()

        plt.subplot(4, 2, 6)
        plt.plot(self.time_vals, self.effective_angle_of_attack_deg_vals, linewidth = 2, label = 'Effective Angle of Attack')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Effective Angle of Attack')
        plt.grid()

        plt.subplot(4, 2, 7)
        plt.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 2, label = 'Flight Path Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Flight Path Angle')
        plt.grid()

        plt.subplot(4, 2, 8)
        plt.plot(self.time_vals, self.u0_vals, linewidth = 2, label = 'RCS Throttle')
        plt.xlabel('Time [s]')
        plt.ylabel('Throttle [-]')
        plt.title('RCS Throttle')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'results/classical_controllers/ballistic_arc_descent.png')
        plt.close()
    
    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        dynamic_pressure = 0.5 * density * math.sqrt(vx**2 + vy**2)**2
        while dynamic_pressure < self.dynamic_pressure_threshold:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
            density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            dynamic_pressure = 0.5 * density * math.sqrt(vx**2 + vy**2)**2
            self.closed_loop_step()
        self.plot_results()
        self.save_results()