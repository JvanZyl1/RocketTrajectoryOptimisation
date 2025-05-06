import os
import csv
import math
import numpy as np
import pandas as pd
from pyswarm import pso
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.envs.base_environment import load_re_entry_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

def throttle_controller(mach_number, air_density, speed_of_sound, Q_max, Kp_mach):
    Q_ref = Q_max - 1000 # [Pa]
    mach_number_max = math.sqrt(2 * Q_ref / air_density) * 1 / speed_of_sound
    error_mach_number = mach_number_max - mach_number
    non_nominal_throttle = np.clip(Kp_mach * error_mach_number, 0, 1) # minimum 40% throttle
    return non_nominal_throttle

def re_entry_burn_pitch_controller(flight_path_angle_rad,
                                   pitch_angle_rad,
                                   previous_alpha_effective_rad,
                                   previous_derivative,
                                   dt,
                                   Kp_pitch,
                                   Kd_pitch,
                                   N_pitch):
    M_max = 0.75e9
    alpha_effective_rad = flight_path_angle_rad - pitch_angle_rad - math.pi
    Mz_command_norm, new_derivative = PD_controller_single_step(Kp=Kp_pitch,
                                                                Kd=Kd_pitch,
                                                                N=N_pitch,
                                                                error=alpha_effective_rad,
                                                                previous_error=previous_alpha_effective_rad,
                                                                previous_derivative=previous_derivative,
                                                                dt=dt)
    Mz = np.clip(Mz_command_norm, -1, 1) * M_max
    return Mz, alpha_effective_rad, new_derivative

def gimbal_determination(Mz,
                         non_nominal_throttle,
                         atmospheric_pressure,
                         d_thrust_cg,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area,
                         max_gimbal_angle_rad,
                         nominal_throttle):
    
    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle

    ratio = -Mz / (thrust_gimballed * d_thrust_cg)
    if ratio > 1:
        gimbal_angle_rad = math.asin(1)
    elif ratio < -1:
        gimbal_angle_rad = math.asin(-1)
    else:
        gimbal_angle_rad = math.asin(ratio)

    gimbal_angle_rad = np.clip(gimbal_angle_rad, -max_gimbal_angle_rad, max_gimbal_angle_rad)

    return gimbal_angle_rad

def augment_actions_re_entry_burn(gimbal_angle_rad, non_nominal_throttle, max_gimbal_angle_rad):
    u0 = gimbal_angle_rad / max_gimbal_angle_rad
    u1 = 2 * non_nominal_throttle - 1
    actions = (u0, u1)
    return actions

def action_determination(mach_number,
                         air_density,
                         speed_of_sound,
                         flight_path_angle_rad,
                         pitch_angle_rad,
                         atmospheric_pressure,
                         d_thrust_cg,
                         previous_alpha_effective_rad,
                         previous_derivative,
                         dt,
                         Q_max,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area,
                         nominal_throttle,
                         max_gimbal_angle_rad,
                         Kp_mach,
                         Kp_pitch,
                         Kd_pitch,
                         N_pitch):
    non_nominal_throttle = throttle_controller(mach_number, air_density, speed_of_sound, Q_max, Kp_mach)
    control_moment_z, alpha_effective_rad, new_derivative = re_entry_burn_pitch_controller(flight_path_angle_rad, pitch_angle_rad, previous_alpha_effective_rad, previous_derivative, dt, Kp_pitch, Kd_pitch, N_pitch)
    gimbal_command_angle_rad = gimbal_determination(control_moment_z, non_nominal_throttle, atmospheric_pressure, d_thrust_cg, number_of_engines_gimballed, thrust_per_engine_no_losses, nozzle_exit_pressure, nozzle_exit_area, max_gimbal_angle_rad, nominal_throttle)
    actions = augment_actions_re_entry_burn(gimbal_command_angle_rad, non_nominal_throttle, max_gimbal_angle_rad)
    info = {
        'gimbal_angle_command_deg': math.degrees(gimbal_command_angle_rad),
        'alpha_effective_rad': alpha_effective_rad,
        'new_derivative': new_derivative,
        'control_moment_z': control_moment_z,
        'non_nominal_throttle': non_nominal_throttle
    }
    return actions, info


class ReEntryBurn_:
    def __init__(self,
                 individual = None):
        self.dt = 0.1
        self.landing_burn_altitude = 5000
        self.max_deflection_angle_deg = 60
        self.Q_max = 30000 # [Pa]
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 're_entry_burn')
        self.action_determination_lambda = self.compile_action_determination_lambda(individual)
        self.initialise_logging()
        self.initial_conditions()

    def compile_action_determination_lambda(self, individual = None):
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]

        number_of_engines_min = 3
        minimum_engine_throttle = 0.4
        nominal_throttle = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
        if individual is not None:
            Kp_pitch, Kd_pitch = individual
            Kp_mach = 0.084
            N_pitch = 14
            self.post_process_results = False
        else:
            Kp_mach = 0.084
            gains = pd.read_csv('data/reference_trajectory/re_entry_burn_controls/gains.csv')
            Kp_pitch = gains['Kp_pitch'].values[0]
            Kd_pitch = gains['Kd_pitch'].values[0]
            N_pitch = 14
            self.post_process_results = True
        action_determination_lambda = lambda mach_number, air_density, speed_of_sound, \
                                     flight_path_angle_rad, pitch_angle_rad, atmospheric_pressure, \
                                     d_thrust_cg, previous_alpha_effective_rad, previous_derivative : action_determination(mach_number=mach_number,
                                                                        air_density=air_density,
                                                                        speed_of_sound=speed_of_sound,
                                                                        flight_path_angle_rad=flight_path_angle_rad,
                                                                        pitch_angle_rad=pitch_angle_rad,
                                                                        atmospheric_pressure=atmospheric_pressure,
                                                                        d_thrust_cg=d_thrust_cg,
                                                                        previous_alpha_effective_rad=previous_alpha_effective_rad,
                                                                        previous_derivative=previous_derivative,
                                                                        dt=self.dt,
                                                                        Q_max=self.Q_max,
                                                                        number_of_engines_gimballed=int(sizing_results['Number of engines gimballed stage 1']),
                                                                        thrust_per_engine_no_losses=float(sizing_results['Thrust engine stage 1']),
                                                                        nozzle_exit_pressure=float(sizing_results['Nozzle exit pressure stage 1']),
                                                                        nozzle_exit_area=float(sizing_results['Nozzle exit area']),
                                                                        nominal_throttle=nominal_throttle,
                                                                        max_gimbal_angle_rad=math.radians(4),
                                                                        Kp_mach=Kp_mach,
                                                                        Kp_pitch=Kp_pitch,
                                                                        Kd_pitch=Kd_pitch,
                                                                        N_pitch=N_pitch)
        return action_determination_lambda

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
        self.non_nominal_throttle_vals = []
        self.throttle_vals = []
        self.gimbal_angle_deg_vals = []
        self.gimbal_angle_command_deg_vals = []

        self.dynamic_pressure_vals = []
        self.control_force_y_vals = []
        self.aero_force_y_vals = []
        self.gravity_force_y_vals = []

        self.control_moment_z_vals = []
        self.aero_moment_z_vals = []
        self.moments_z_vals = []

        self.air_densities = []
        self.speed_of_sounds = []

    def initial_conditions(self):
        self.state = load_re_entry_burn_initial_state('supervisory')
        self.previous_alpha_effective_rad = 0.0
        self.previous_derivative = 0.0

        self.y0 = self.state[1]
        self.air_density, self.atmospheric_pressure, self.speed_of_sound = endo_atmospheric_model(self.y0)
        speed = math.sqrt(self.state[2]**2 + self.state[3]**2)
        self.mach_number = speed / self.speed_of_sound
        self.dynamic_pressure = 0.5 * self.air_density * speed**2
        _, info_0 = self.simulation_step_lambda(self.state, (0,0),0.0, 0.0, None)
        self.d_thrust_cg = info_0['d_thrust_cg']

        self.gimbal_angle_deg_prev = 0.0
        self.deflection_angle_rad_prev = 0.0

    def reset(self):
        self.initial_conditions()
        self.initialise_logging()

    def dynamic_pressure_constraint(self):
        return self.Q_max - max(self.dynamic_pressure_vals)

    def performance_metrics(self):
        # massive penalty is dynamic pressure of Qmax
        reward = 0
        # Next aim minimise alpha effective
        for i in range(len(self.effective_angle_of_attack_deg_vals)):
            alpha_effective_error = abs(self.effective_angle_of_attack_deg_vals[i])
            reward -= alpha_effective_error/1e2
        reward -= abs(max(self.effective_angle_of_attack_deg_vals[35:]))
        reward -= abs(math.degrees(self.effective_angle_of_attack_deg_vals[-1]))*5/2
        return reward
    
    def closed_loop_step(self):
        actions, actions_info = self.action_determination_lambda(mach_number=self.mach_number,
                                                                  air_density=self.air_density,
                                                                  speed_of_sound=self.speed_of_sound,
                                                                  flight_path_angle_rad=self.state[6],
                                                                  pitch_angle_rad=self.state[4],
                                                                  atmospheric_pressure=self.atmospheric_pressure,
                                                                  d_thrust_cg=self.d_thrust_cg,
                                                                  previous_alpha_effective_rad=self.previous_alpha_effective_rad,
                                                                  previous_derivative=self.previous_derivative)
        self.previous_alpha_effective_rad = actions_info['alpha_effective_rad']
        self.previous_derivative = actions_info['new_derivative']
        self.state, info = self.simulation_step_lambda(self.state, actions, self.gimbal_angle_deg_prev, self.deflection_angle_rad_prev, None)
        self.air_density, self.speed_of_sound, self.mach_number = info['air_density'], info['speed_of_sound'], info['mach_number']
        self.atmospheric_pressure = info['atmospheric_pressure']
        self.d_thrust_cg = info['d_thrust_cg']
        self.gimbal_angle_deg_prev = info['action_info']['gimbal_angle_deg']
        self.deflection_angle_rad_prev = info['action_info']['deflection_angle_rad']
        
        self.dynamic_pressure = info['dynamic_pressure']
        self.throttle_vals.append(info['action_info']['throttle'])

        self.air_densities.append(self.air_density)
        self.speed_of_sounds.append(self.speed_of_sound)    

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
        self.non_nominal_throttle_vals.append(actions_info['non_nominal_throttle'])
        self.gimbal_angle_command_deg_vals.append(actions_info['gimbal_angle_command_deg'])
        self.gimbal_angle_deg_vals.append(info['action_info']['gimbal_angle_deg'])

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
            'non_nominal_throttle': self.non_nominal_throttle_vals,
            'gimbalanglecommand[deg]': self.gimbal_angle_command_deg_vals,
            'gimbalangle[deg]': self.gimbal_angle_deg_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals,
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
        ax6.plot(self.time_vals, self.gimbal_angle_deg_vals, linewidth = 4, color = 'blue', label = 'Gimbal Angle')
        ax6.plot(self.time_vals, np.clip(np.array(self.gimbal_angle_command_deg_vals),
                                          min(self.gimbal_angle_deg_vals),
                                          max(self.gimbal_angle_deg_vals)), linewidth = 3, alpha = 0.5, linestyle = '--', color = 'pink', label = 'Gimbal Angle Command')
        ax6.set_xlabel('Time [s]', fontsize=20)
        ax6.set_ylabel(r'Angle [$^\circ$]', fontsize=20)
        ax6.set_title('Gimbal Angle', fontsize=22)
        ax6.grid()
        ax6.legend(fontsize=16)
        ax6.tick_params(axis='both', which='major', labelsize=16)

        plt.savefig(f'results/classical_controllers/re_entry_burn_rotational_motion.png')
        plt.close()

    def run_closed_loop(self):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        while vx < -4.0:
            self.closed_loop_step()
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        if self.post_process_results:
            self.plot_results()
            self.save_results()


def objective_func_lambda(individual):
    re_entry_burn = ReEntryBurn_(individual)
    re_entry_burn.run_closed_loop()
    obj = -re_entry_burn.performance_metrics()
    print('Objective: ', obj)
    return obj

def constraint_func_lambda(individual):
    re_entry_burn = ReEntryBurn_(individual)
    re_entry_burn.run_closed_loop()
    max_pressure = max(re_entry_burn.dynamic_pressure_vals)
    cons = re_entry_burn.Q_max - max_pressure
    # PSO expects constraints in the form of >= 0
    return [cons]

def save_gains(xopt):
    xopt = np.array(xopt)
    print(xopt)
    # Creating DataFrame with index [0] to avoid "ValueError: If using all scalar values, you must pass an index"
    gains = pd.DataFrame({'Kp_pitch': [xopt[0]], 'Kd_pitch': [xopt[1]]})
    gains.to_csv('data/reference_trajectory/re_entry_burn_controls/gains.csv', index=False)

def tune_re_entry_burn():
    # Kp_pitch, Kd_pitch
    lb = [0, 0]  # Lower bounds
    ub = [5, 5]     # Upper bounds
    
    xopt, fopt = pso(
        objective_func_lambda,
        lb, 
        ub,
        f_ieqcons=constraint_func_lambda,
        swarmsize=30,      # Number of particles
        omega=0.5,         # Particle velocity scaling factor
        phip=0.5,          # Scaling factor for particle's best known position
        phig=0.5,          # Scaling factor for swarm's best known position
        maxiter=40,        # Maximum iterations
        minstep=1e-6,      # Minimum step size before search termination
        minfunc=1e-6,      # Minimum change in obj value before termination
        debug=True,        # Print progress statements
    )
    save_gains(xopt)
    return ReEntryBurn_(xopt)

def compile_re_entry_burn(tune_bool = False):
    if not tune_bool:
        return ReEntryBurn_()
    else:
        return tune_re_entry_burn()

class ReEntryBurn:
    def __init__(self, tune_bool = False):
        self.env = compile_re_entry_burn(tune_bool = tune_bool)

    def run_closed_loop(self):
        self.env.run_closed_loop()

    def plot_results(self):
        self.env.plot_results()

    def save_results(self):
        self.env.save_results()