import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.envs.base_environment import load_re_entry_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.classical_controls.re_entry_burn_gain_schedule import solve_gain_schedule

def throttle_controller(mach_number, air_density, speed_of_sound, Q_max):
    Kp_mach = 0.065
    Q_ref = Q_max - 1000 # [Pa]
    mach_number_max = math.sqrt(2 * Q_ref / air_density) * 1 / speed_of_sound
    error_mach_number = mach_number_max - mach_number
    non_nominal_throttle = np.clip(Kp_mach * error_mach_number, 0, 1) # minimum 40% throttle
    return non_nominal_throttle

def ACS_controller(state,
                   dynamic_pressure,
                   previous_alpha_effective_rad,
                   previous_derivative,
                   max_deflection_angle_deg,
                   dt,
                   gains_ACS_re_entry_burn):
    x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
    alpha_effective_rad = gamma - theta - math.pi
    # Define gain schedules for increasing and decreasing dynamic pressure
    Kp_alpha_ballistic_arc, Kd_alpha_ballistic_arc = gains_ACS_re_entry_burn
        
    # Apply PD control
    delta_norm, new_derivative = PD_controller_single_step(Kp=Kp_alpha_ballistic_arc,
                                                           Kd=Kd_alpha_ballistic_arc,
                                                           N=30,
                                                           error=alpha_effective_rad,
                                                           previous_error=previous_alpha_effective_rad,
                                                           previous_derivative=previous_derivative,
                                                           dt=dt)
    
    # Clip control output
    delta_norm = np.clip(delta_norm, -1, 1)
    
    # Convert to degrees
    delta_left_deg = delta_norm * max_deflection_angle_deg 
    delta_right_deg = delta_left_deg

    return delta_left_deg, delta_right_deg, alpha_effective_rad, new_derivative

def augment_action_ACS(delta_left_deg, delta_right_deg, max_deflection_angle_deg):
    u0 = delta_left_deg / max_deflection_angle_deg
    u1 = delta_right_deg / max_deflection_angle_deg
    return u0, u1

def augment_action_throttle(non_nominal_throttle):
    u2 = 2 * non_nominal_throttle - 1
    return u2

class ReEntryBurn:
    def __init__(self,
                 tune_ACS_bool = False):
        self.dt = 0.1
        self.landing_burn_altitude = 5000
        self.max_deflection_angle_deg = 60
        self.Q_max = 30000 # [Pa]
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 're_entry_burn')
        
        if tune_ACS_bool:
            self.gains_ACS_re_entry_burn = solve_gain_schedule()
        else:
            # file path : data/reference_trajectory/re_entry_burn_controls/ACS_re_entry_burn_gain_schedule.csv
            self.gains_ACS_re_entry_burn = pd.read_csv('data/reference_trajectory/re_entry_burn_controls/ACS_re_entry_burn_gain_schedule.csv')
            self.gains_ACS_re_entry_burn = self.gains_ACS_re_entry_burn.values[0]
            self.gains_ACS_re_entry_burn = [-9, 0.0]
        self.acs_controller_lambda = lambda state, dynamic_pressure, previous_alpha_effective_rad, previous_derivative: ACS_controller(state,
                                                                                                                     dynamic_pressure,
                                                                                                                     previous_alpha_effective_rad,
                                                                                                                     previous_derivative,
                                                                                                                     max_deflection_angle_deg = self.max_deflection_angle_deg,
                                                                                                                     dt = self.dt,
                                                                                                                     gains_ACS_re_entry_burn = self.gains_ACS_re_entry_burn)
        
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
        self.non_nominal_throttle_vals = []
        self.throttle_vals = []
        self.delta_left_deg_vals = []
        self.delta_right_deg_vals = []

        self.dynamic_pressure_vals = []
        self.control_force_y_vals = []
        self.aero_force_y_vals = []
        self.gravity_force_y_vals = []

        self.control_moment_z_vals = []
        self.aero_moment_z_vals = []
        self.moments_z_vals = []
        self.t_gain_switch_vals = []
        self.previous_gain_dp = 1

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
        non_nominal_throttle = self.throttle_controller_lambda(self.mach_number, self.air_density, self.speed_of_sound)
        u2 = self.augment_action_throttle_lambda(non_nominal_throttle)
        actions = (u0, u1, u2)

        self.state, info = self.simulation_step_lambda(self.state, actions, self.delta_left_deg_prev, self.delta_right_deg_prev)
        self.delta_left_deg_prev, self.delta_right_deg_prev = info['action_info']['delta_left_deg'], info['action_info']['delta_right_deg']
        self.air_density, self.speed_of_sound, self.mach_number = info['air_density'], info['speed_of_sound'], info['mach_number']
        self.dynamic_pressure = info['dynamic_pressure']
        self.throttle_vals.append(info['action_info']['throttle'])

        # Gain schedule tracking
        if self.dynamic_pressure > 0 and self.dynamic_pressure < 5000:
            gain_dp = 1
        elif self.dynamic_pressure > 5000 and self.dynamic_pressure < 10000:
            gain_dp = 2
        elif self.dynamic_pressure > 10000 and self.dynamic_pressure < 15000:
            gain_dp = 3
        elif self.dynamic_pressure > 15000 and self.dynamic_pressure < 20000:
            gain_dp = 4
        elif self.dynamic_pressure > 20000 and self.dynamic_pressure < 25000:
            gain_dp = 5
        else: # 25000 > dynamic_pressure
            gain_dp = 6
        if gain_dp != self.previous_gain_dp:
            self.t_gain_switch_vals.append(self.state[-1])
            self.previous_gain_dp = gain_dp
            # Gain schedule tracking
                

        self.previous_dynamic_pressure_val = self.dynamic_pressure

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
        self.non_nominal_throttle_vals.append(non_nominal_throttle)
        self.delta_left_deg_vals.append(info['action_info']['delta_left_deg'])
        self.delta_right_deg_vals.append(info['action_info']['delta_right_deg'])

        self.dynamic_pressure_vals.append(info['dynamic_pressure'])
        self.control_force_y_vals.append(info['control_force_y'])
        self.aero_force_y_vals.append(info['aero_force_y'])
        self.gravity_force_y_vals.append(info['gravity_force_y'])

        self.control_moment_z_vals.append(info['moment_dict']['control_moment_z'])
        self.aero_moment_z_vals.append(info['moment_dict']['aero_moment_z'])
        self.moments_z_vals.append(info['moment_dict']['moments_z'])

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
            'non_nominal_throttle': self.non_nominal_throttle_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals,
            'u2': self.u2_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_re_entry_burn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        self.total_force_y_vals = np.array(self.control_force_y_vals) + np.array(self.aero_force_y_vals) + np.array(self.gravity_force_y_vals)
        pitch_angle_down_deg_vals = np.array(self.pitch_angle_deg_vals) + 180
        # Set default font sizes
        plt.rcParams.update({
            'font.size': 12,           # Default font size
            'axes.titlesize': 14,      # Title font size
            'axes.labelsize': 12,      # Axis label font size
            'xtick.labelsize': 10,     # X-axis tick label size
            'ytick.labelsize': 10,     # Y-axis tick label size
            'legend.fontsize': 10      # Legend font size
        })

        # A4 size plot
        plt.figure(figsize=(20, 15))
        plt.suptitle('Re-entry Burn', fontsize=32)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(np.array(self.x_vals)/1000, np.array(self.y_vals)/1000, linewidth = 4, color = 'blue', label = 'Flight Path')
        # Start and end points
        ax1.scatter(self.x_vals[0]/1000, self.y_vals[0]/1000, color = 'green', marker = 'o', s = 100, label = 'Start')
        ax1.scatter(self.x_vals[-1]/1000, self.y_vals[-1]/1000, color = 'red', marker = 'o', s = 100, label = 'End')
        ax1.set_xlabel('x [km]', fontsize=20)
        ax1.set_ylabel('y [km]', fontsize=20)
        ax1.set_title('Flight Path', fontsize=22)
        ax1.grid()
        ax1.legend(fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, np.array(self.dynamic_pressure_vals)/1000, linewidth = 4, color = 'blue')
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_ylabel('Dynamic Pressure [kPa]', fontsize=20)
        ax2.set_title('Dynamic Pressure', fontsize=22)
        ax2.grid()
        ax2.tick_params(axis='both', which='major', labelsize=16)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, self.vx_vals, linewidth = 4, color = 'blue')
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.set_ylabel('Velocity [m/s]', fontsize=20)
        ax3.set_title('Velocity x', fontsize=22)
        ax3.grid()
        ax3.tick_params(axis='both', which='major', labelsize=16)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.vy_vals, linewidth = 4, color = 'blue')
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.set_ylabel('Velocity [m/s]', fontsize=20)
        ax4.set_title('Velocity y', fontsize=22)
        ax4.grid()
        ax4.tick_params(axis='both', which='major', labelsize=16)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, self.throttle_vals, linewidth = 4, color = 'blue')
        ax5.set_xlabel('Time [s]', fontsize=20)
        ax5.set_ylabel('Throttle', fontsize=20)
        ax5.set_title('Throttle (40% minimum throttle)', fontsize=22)
        ax5.grid()
        ax5.tick_params(axis='both', which='major', labelsize=16)

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, np.array(self.total_force_y_vals)/1e6, label = 'Total Force y', linewidth = 5.5, color = 'blue', linestyle = '--')
        ax6.plot(self.time_vals, np.array(self.control_force_y_vals)/1e6, label = 'Control Force y', linewidth = 3, color = 'cyan')
        ax6.plot(self.time_vals, np.array(self.aero_force_y_vals)/1e6, label = 'Aero Force y', linewidth = 3, color = 'orange')
        ax6.plot(self.time_vals, np.array(self.gravity_force_y_vals)/1e6, label = 'Gravity Force y', linewidth = 3, color = 'green')
        ax6.set_xlabel('Time [s]', fontsize=20)
        ax6.set_ylabel('Force [MN]', fontsize=20)
        ax6.set_title('Forces - y', fontsize=22)
        ax6.grid()
        ax6.legend(fontsize=16)
        ax6.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(f'results/classical_controllers/re_entry_burn_dynamics.png')
        plt.close()

        plt.figure(figsize=(20, 15))
        plt.suptitle('Re-entry Burn', fontsize=32)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_vals, pitch_angle_down_deg_vals, linewidth = 4, color = 'blue', label = 'Pitch Angle (Down)')
        ax1.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 3, linestyle = '--', color = 'red', label = 'Flight Path Angle')
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel(r'Angle [$^\circ$]', fontsize=20)
        ax1.set_title('Angles', fontsize=22)
        ax1.grid()
        ax1.legend(fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, self.effective_angle_of_attack_deg_vals, linewidth = 4, color = 'blue', label = 'Effective Angle of Attack')
        ax2.plot(self.time_vals, np.zeros_like(self.time_vals), linewidth = 3, linestyle = '--', color = 'red', label = 'Zero AOA')
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_ylabel(r'Angle [$^\circ$]', fontsize=20)
        ax2.set_title('Effective Angle of Attack', fontsize=22)
        ax2.grid()
        ax2.legend(fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, np.array(self.control_moment_z_vals)/1e6, linewidth = 3, color = 'orange', label = 'Control Moment z')
        ax3.plot(self.time_vals, np.array(self.aero_moment_z_vals)/1e6, linewidth = 3, color = 'green', label = 'Aero Moment z')
        ax3.plot(self.time_vals, np.array(self.moments_z_vals)/1e6, linewidth = 4, linestyle = '--', color = 'blue', label = 'Total Moment z')
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.set_ylabel('Moment [MNm]', fontsize=20)
        ax3.set_title('Moments', fontsize=22)
        ax3.grid()
        ax3.legend(fontsize=16)
        ax3.tick_params(axis='both', which='major', labelsize=16)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, np.array(self.dynamic_pressure_vals)/1000, linewidth = 4, color = 'blue')
        for t_gain_switch in self.t_gain_switch_vals:
            ax4.axvline(x=t_gain_switch, color='blue', linestyle='--', linewidth = 1.5, label='Gain Switch')
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.set_ylabel('Dynamic Pressure [kPa]', fontsize=20)
        ax4.set_title('Dynamic Pressure', fontsize=22)
        ax4.grid()
        ax4.tick_params(axis='both', which='major', labelsize=16)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, self.pitch_rate_deg_vals, linewidth = 4, color = 'blue', label = 'Pitch Rate')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero Rate')
        ax5.set_xlabel('Time [s]', fontsize=20)
        ax5.set_ylabel(r'Pitch Rate [$^\circ$/s]', fontsize=20)
        ax5.set_title('Pitch Rate', fontsize=22)
        ax5.grid()
        ax5.tick_params(axis='both', which='major', labelsize=16)

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, self.delta_left_deg_vals, linewidth = 4, color = 'blue', label = 'delta_left_deg')
        for t_gain_switch in self.t_gain_switch_vals:
            ax6.axvline(x=t_gain_switch, color='blue', linestyle='--', linewidth = 1.5, label='Gain Switch')
        ax6.set_xlabel('Time [s]', fontsize=20)
        ax6.set_ylabel(r'Angle [$^\circ$]', fontsize=20)
        ax6.set_title('Grid fin\'s deflection angles', fontsize=22)
        ax6.grid()
        ax6.tick_params(axis='both', which='major', labelsize=16)

        plt.savefig(f'results/classical_controllers/re_entry_burn_rotational_motion.png')
        plt.close()

    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        while vx < -0.1:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        print(f'Final states: x: {x}, y: {y}, vx: {vx}, vy: {vy}')
        self.plot_results()
        self.save_results()  