import os
import math
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.rockets_physics import compile_physics
from src.envs.base_environment import load_subsonic_initial_state

def ascent_reference_pitch(time, T_final):
    pitch_ref_deg = 90 - 35 / (1 + np.exp(-0.05 * (time - 6/9 * T_final)))
    return math.radians(pitch_ref_deg)

def ascent_pitch_controller(pitch_reference_rad,
                            pitch_angle_rad):
    Kp_pitch = 0.61
    error_pitch_angle = pitch_reference_rad - pitch_angle_rad
    M_max = 0.75e9
    Mz = np.clip(Kp_pitch * error_pitch_angle, -1, 1) * M_max
    return Mz

def ascent_controller_step(mach_number_reference_previous,
                           mach_number,
                           air_density,
                           speed_of_sound):
    Kp_mach = 20
    Q_max = 30000 # [Pa]
    if speed_of_sound > 0.1:
        mach_number_max = math.sqrt(2 * Q_max / air_density) * 1 / speed_of_sound
        mach_reference_rl = 0.2
        mach_number_reference = max(mach_number_reference_previous - mach_reference_rl, min(mach_number_reference_previous + mach_reference_rl, mach_number_max))
        error_mach_number = mach_number_reference - mach_number
        throttle_non_nom = np.clip(Kp_mach * error_mach_number, -1, 1)
    else:
        throttle_non_nom = 1.0
        mach_number_reference = 4.0

    return throttle_non_nom, mach_number_reference

def gimbal_determination(Mz,
                         non_nominal_throttle,
                         atmospheric_pressure,
                         d_thrust_cg,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area,
                         nominal_throttle = 0.5):
    
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

    return gimbal_angle_rad

def augment_actions_ascent_control(gimbal_angle_rad, non_nominal_throttle, max_gimbal_angle_rad):
    u0 = gimbal_angle_rad / max_gimbal_angle_rad
    u1 = 2 * non_nominal_throttle - 1

    actions = (u0, u1)
    return actions

class AscentControl:
    def __init__(self):
        self.T_final = 120
        self.dt = 0.1
        self.max_gimbal_angle_rad = math.radians(1)
        self.nominal_throttle = 0.5

        self.augment_actions_lambda = lambda gimbal_angle_rad, non_nominal_throttle : augment_actions_ascent_control(gimbal_angle_rad, non_nominal_throttle, self.max_gimbal_angle_rad)
        self.pitch_reference_lambda = lambda time : ascent_reference_pitch(time, self.T_final)

        self.state = load_subsonic_initial_state()
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'subsonic')
        
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]

        self.burn_out_mass = float(sizing_results['Ascent burnout mass (subrocket 0)'])*1000
        
        self.gimbal_determiner = lambda Mz, non_nominal_throttle, atmospheric_pressure, d_thrust_cg : gimbal_determination(
            Mz, non_nominal_throttle, atmospheric_pressure, d_thrust_cg,
            number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1']),
            thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
            nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
            nozzle_exit_area = float(sizing_results['Nozzle exit area']),
            nominal_throttle = self.nominal_throttle
        )

        self.mach_number_reference_previous = 0.0
        self.initial_conditions()
        self.initialise_logging()

    def reset(self):
        self.initial_conditions()
        self.initialise_logging()
        self.mach_number_reference_previous = 0.0


    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.pitch_angle_deg_vals = []
        self.pitch_angle_reference_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.mach_number_reference_vals = []
        self.angle_of_attack_deg_vals = []
        self.gimbal_angle_deg_vals = []
        self.non_nominal_throttle_vals = []
        self.u0_vals = []
        self.u1_vals = []

        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.mass_propellant_vals = []

    def initial_conditions(self):
        _, info_IC = self.simulation_step_lambda(self.state, (0,0), None)
        self.atmospheric_pressure = info_IC['atmospheric_pressure']
        self.air_density = info_IC['air_density']
        self.speed_of_sound = info_IC['speed_of_sound']
        self.mach_number = 0.0
        self.d_thrust_cg = info_IC['d_thrust_cg']
        self.time = 0.0
        self.pitch_angle_rad = math.pi/2
        self.gamma_rad = math.pi/2

    def closed_loop_step(self):
        pitch_reference_rad = self.pitch_reference_lambda(self.time)
        control_moments =  ascent_pitch_controller(pitch_reference_rad, self.pitch_angle_rad)
        non_nominal_throttle, self.mach_number_reference_previous = ascent_controller_step(self.mach_number_reference_previous,
                           self.mach_number,
                           self.air_density,
                           self.speed_of_sound)
        
        gimbal_angle_rad = self.gimbal_determiner(control_moments, non_nominal_throttle, self.atmospheric_pressure, self.d_thrust_cg)        
        actions = self.augment_actions_lambda(gimbal_angle_rad, non_nominal_throttle)

        self.state, info = self.simulation_step_lambda(self.state, actions, None)
        
        # Update local variables
        self.atmospheric_pressure = info['atmospheric_pressure']
        self.air_density = info['air_density']
        self.speed_of_sound = info['speed_of_sound']
        self.mach_number = info['mach_number']
        self.d_thrust_cg = info['d_thrust_cg']
        self.time = self.state[-1]
        self.pitch_angle_rad = self.state[4]

        # Do logging
        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.pitch_angle_rad))
        self.pitch_angle_reference_deg_vals.append(math.degrees(pitch_reference_rad))
        self.time_vals.append(self.time)
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(self.mach_number)
        self.mach_number_reference_vals.append(self.mach_number_reference_previous)
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.gimbal_angle_deg_vals.append(math.degrees(gimbal_angle_rad))
        self.non_nominal_throttle_vals.append(non_nominal_throttle)
        self.u0_vals.append(actions[0])
        self.u1_vals.append(actions[1])
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])
        self.mass_propellant_vals.append(self.state[9])

    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/ascent_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_ascent_control.csv')
        subsonic_trajectory_path = os.path.join(save_folder, 'subsonic_trajectory_ascent_control.csv')
        supersonic_trajectory_path = os.path.join(save_folder, 'supersonic_trajectory_ascent_control.csv')

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

        # Create subsonic and supersonic DataFrames
        subsonic_data = {
            't[s]': [t for t, mach in zip(self.time_vals, self.mach_number_vals) if mach < 1],
            'x[m]': [x for x, mach in zip(self.x_vals, self.mach_number_vals) if mach < 1],
            'y[m]': [y for y, mach in zip(self.y_vals, self.mach_number_vals) if mach < 1],
            'vx[m/s]': [vx for vx, mach in zip(self.vx_vals, self.mach_number_vals) if mach < 1],
            'vy[m/s]': [vy for vy, mach in zip(self.vy_vals, self.mach_number_vals) if mach < 1],
            'mass[kg]': [mass for mass, mach in zip(self.mass_vals, self.mach_number_vals) if mach < 1]
        }

        supersonic_data = {
            't[s]': [t for t, mach in zip(self.time_vals, self.mach_number_vals) if mach >= 1],
            'x[m]': [x for x, mach in zip(self.x_vals, self.mach_number_vals) if mach >= 1],
            'y[m]': [y for y, mach in zip(self.y_vals, self.mach_number_vals) if mach >= 1],
            'vx[m/s]': [vx for vx, mach in zip(self.vx_vals, self.mach_number_vals) if mach >= 1],
            'vy[m/s]': [vy for vy, mach in zip(self.vy_vals, self.mach_number_vals) if mach >= 1],
            'mass[kg]': [mass for mass, mach in zip(self.mass_vals, self.mach_number_vals) if mach >= 1]
        }

        pd.DataFrame(subsonic_data).to_csv(subsonic_trajectory_path, index=False)
        pd.DataFrame(supersonic_data).to_csv(supersonic_trajectory_path, index=False)

        # Now state action variables
        # time,x,y,vx,vy,theta,theta_dot,alpha,mass,
        # time[s],x[m],y[m],vx[m/s],vy[m/s],theta[rad],theta_dot[rad/s],alpha[rad],mass[kg],gimbalangle[rad],nonnominalthrottle[0-1]
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
            'gimbalangle[deg]': self.gimbal_angle_deg_vals,
            'nonnominalthrottle[0-1]': self.non_nominal_throttle_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_ascent_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

        # Now state action variables for subsonic and supersonic
        subsonic_state_action_data = {
            'time[s]': [t for t, mach in zip(self.time_vals, self.mach_number_vals) if mach < 1],
            'x[m]': [x for x, mach in zip(self.x_vals, self.mach_number_vals) if mach < 1],
            'y[m]': [y for y, mach in zip(self.y_vals, self.mach_number_vals) if mach < 1],
            'vx[m/s]': [vx for vx, mach in zip(self.vx_vals, self.mach_number_vals) if mach < 1],
            'vy[m/s]': [vy for vy, mach in zip(self.vy_vals, self.mach_number_vals) if mach < 1],
            'theta[rad]': [theta for theta, mach in zip(pitch_angle_rad_vals, self.mach_number_vals) if mach < 1],
            'theta_dot[rad/s]': [theta_dot for theta_dot, mach in zip(pitch_rate_rad_vals, self.mach_number_vals) if mach < 1],
            'alpha[rad]': [alpha for alpha, mach in zip(angle_of_attack_rad_vals, self.mach_number_vals) if mach < 1],
            'mass[kg]': [mass for mass, mach in zip(self.mass_vals, self.mach_number_vals) if mach < 1],
            'gimbalangle[deg]': [gimbal_angle for gimbal_angle, mach in zip(self.gimbal_angle_deg_vals, self.mach_number_vals) if mach < 1],
            'nonnominalthrottle[0-1]': [non_nominal_throttle for non_nominal_throttle, mach in zip(self.non_nominal_throttle_vals, self.mach_number_vals) if mach < 1],
            'u0': [u0 for u0, mach in zip(self.u0_vals, self.mach_number_vals) if mach < 1],
            'u1': [u1 for u1, mach in zip(self.u1_vals, self.mach_number_vals) if mach < 1]
        }

        supersonic_state_action_data = {
            'time[s]': [t for t, mach in zip(self.time_vals, self.mach_number_vals) if mach >= 1],
            'x[m]': [x for x, mach in zip(self.x_vals, self.mach_number_vals) if mach >= 1],
            'y[m]': [y for y, mach in zip(self.y_vals, self.mach_number_vals) if mach >= 1],
            'vx[m/s]': [vx for vx, mach in zip(self.vx_vals, self.mach_number_vals) if mach >= 1],
            'vy[m/s]': [vy for vy, mach in zip(self.vy_vals, self.mach_number_vals) if mach >= 1],
            'theta[rad]': [math.radians(theta) for theta, mach in zip(self.pitch_angle_deg_vals, self.mach_number_vals) if mach >= 1],
            'theta_dot[rad/s]': [math.radians(theta_dot) for theta_dot, mach in zip(self.pitch_rate_deg_vals, self.mach_number_vals) if mach >= 1],
            'alpha[rad]': [math.radians(alpha) for alpha, mach in zip(self.angle_of_attack_deg_vals, self.mach_number_vals) if mach >= 1],
            'mass[kg]': [mass for mass, mach in zip(self.mass_vals, self.mach_number_vals) if mach >= 1],
            'gimbalangle[deg]': [gimbal_angle for gimbal_angle, mach in zip(self.gimbal_angle_deg_vals, self.mach_number_vals) if mach >= 1],
            'nonnominalthrottle[0-1]': [non_nominal_throttle for non_nominal_throttle, mach in zip(self.non_nominal_throttle_vals, self.mach_number_vals) if mach >= 1],
            'u0': [u0 for u0, mach in zip(self.u0_vals, self.mach_number_vals) if mach >= 1],
            'u1': [u1 for u1, mach in zip(self.u1_vals, self.mach_number_vals) if mach >= 1]
        }

        subsonic_state_action_path = os.path.join(save_folder, 'subsonic_state_action_ascent_control.csv')
        pd.DataFrame(subsonic_state_action_data).to_csv(subsonic_state_action_path, index=False)

        supersonic_state_action_path = os.path.join(save_folder, 'supersonic_state_action_ascent_control.csv')
        pd.DataFrame(supersonic_state_action_data).to_csv(supersonic_state_action_path, index=False)

    def calculate_velocity_increment(self):
        xt, yt, vxt, vyt, thetat, theta_dott, gammat, alphat, masst, mass_propellant_t, time_t = self.state   
        delta_v_a_1 = np.sqrt(vxt**2 + vyt**2)
        # read csv
        with open('data/rocket_parameters/velocity_increments.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '(sizing output) dv_star_a_1':
                    delta_v_a_1_star = float(row[1])
                if row[0] == '(sizing input) dv_loss_a_1':
                    delta_v_a_1_loss_prev = float(row[1])

        # Calculate delta_v_a_loss
        delta_v_a_1_loss_new = delta_v_a_1 - delta_v_a_1_star

        delta_v_a_1_loss_error = delta_v_a_1_loss_new - delta_v_a_1_loss_prev

        return delta_v_a_1, delta_v_a_1_loss_new, delta_v_a_1_loss_error


        
    def plot_results(self):
        # A4 size plot
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.5, wspace=0.3)
        plt.suptitle('Ascent Control', fontsize = 32)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(np.array(self.x_vals)/1000, np.array(self.y_vals)/1000, linewidth = 4, color = 'blue')
        ax1.set_xlabel('x [km]', fontsize = 20)
        ax1.set_ylabel('y [km]', fontsize = 20)
        ax1.set_title('Flight Path', fontsize = 22)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid()

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, self.mach_number_vals, linewidth = 4, color = 'blue', label = 'Mach Number')
        ax2.plot(self.time_vals, self.mach_number_reference_vals, linewidth = 4, color = 'red', linestyle = '--', label = 'Mach Number Reference')
        ax2.set_xlabel('Time [s]', fontsize = 20)
        ax2.set_ylabel('Mach [-]', fontsize = 20)
        ax2.set_title('Mach Number', fontsize = 22)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.set_ylim(0, 4.0)
        ax2.grid()
        ax2.legend(fontsize = 16)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, self.pitch_angle_deg_vals, linewidth = 3, color = 'blue', label = 'Pitch Angle')
        ax3.plot(self.time_vals, self.pitch_angle_reference_deg_vals, linestyle = '--', linewidth = 2, color = 'red', label = 'Pitch Angle Reference')
        ax3.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 3, color = 'green', label = 'Flight Path Angle')
        ax3.set_xlabel('Time [s]', fontsize = 20)
        ax3.set_ylabel(r'Angle [$^\circ$]', fontsize = 20)
        ax3.set_title('Pitch and Flight Path Angles', fontsize = 22)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.grid()
        ax3.legend(fontsize = 16)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.angle_of_attack_deg_vals, linewidth = 4, label = 'Angle of Attack', color = 'blue')
        ax4.set_xlabel('Time [s]', fontsize = 20)
        ax4.set_ylabel(r'Angle [$^\circ$]', fontsize = 20)
        ax4.set_title('Angle of Attack', fontsize = 22)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.grid()

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, self.gimbal_angle_deg_vals, linewidth = 4, label = 'Gimbal Angle', color = 'blue')
        ax5.set_xlabel('Time [s]', fontsize = 20)
        ax5.set_ylabel(r'Angle [$^\circ$]', fontsize = 20)
        ax5.set_title('Gimbal Angle', fontsize = 22)
        ax5.tick_params(axis='both', which='major', labelsize=16)
        ax5.grid()

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, self.non_nominal_throttle_vals, linewidth = 4, label = 'Non Nominal Throttle', color = 'blue')
        ax6.set_xlabel('Time [s]', fontsize = 20)
        ax6.set_ylabel('Throttle [0-1]', fontsize = 20)
        ax6.set_title('Non-nominal Throttle', fontsize = 22)
        ax6.tick_params(axis='both', which='major', labelsize=16)
        ax6.grid()

        ax7 = plt.subplot(gs[3, 0])
        ax7.plot(self.time_vals, self.mass_propellant_vals, linewidth = 4, label = 'Mass', color = 'blue')
        ax7.set_xlabel('Time [s]', fontsize = 20)
        ax7.set_ylabel('Mass [kg]', fontsize = 20)
        ax7.set_title('Propellant Mass', fontsize = 22)
        ax7.tick_params(axis='both', which='major', labelsize=16)
        ax7.grid()

        ax8 = plt.subplot(gs[3, 1])
        ax8.plot(self.time_vals, self.mass_vals, linewidth = 4, label = 'Mass', color = 'blue')
        ax8.set_xlabel('Time [s]', fontsize = 20)
        ax8.set_ylabel('Mass [kg]', fontsize = 20)
        ax8.set_title('Mass', fontsize = 22)
        ax8.tick_params(axis='both', which='major', labelsize=16)
        ax8.grid()

        plt.savefig(f'results/classical_controllers/endo_ascent.png')
        plt.close()

        delta_v_a_1, delta_v_a_1_loss, delta_v_a_1_loss_error = self.calculate_velocity_increment()
        print(f'Delta V a1: {delta_v_a_1} m/s')
        # save to csv
        with open('data/rocket_parameters/velocity_increments.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['(controller results) delta_v_a_1', delta_v_a_1])
            writer.writerow(['(controller results) delta_v_a_1_loss', delta_v_a_1_loss])
            writer.writerow(['(controller results) delta_v_a_1_loss_error', delta_v_a_1_loss_error])

    def run_closed_loop(self):
        while self.state[-1] < self.T_final and self.state[8] > self.burn_out_mass:
            self.closed_loop_step()
        print(f'Stopped at time {self.state[-1]} s and mass {self.state[8]} kg')
        self.plot_results()
        self.save_results()
if __name__ == "__main__":
    ascent_control = AscentControl()
    ascent_control.run_closed_loop()