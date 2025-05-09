import os
import math
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.rockets_physics import compile_physics
from src.envs.base_environment import load_landing_burn_initial_state
from src.classical_controls.utils import PD_controller_single_step

def throttle_controllers(v_y, v_ref, previous_error, previous_derivative, dt):
    Kp_throttle = 0.8
    Kd_throttle = 0.01
    N_throttle = 10
    error = abs(v_y) - v_ref
    throttle, new_derivative = PD_controller_single_step(Kp=Kp_throttle,
                                                                         Kd=Kd_throttle,
                                                                         N=N_throttle,
                                                                         error=error,
                                                                         previous_error=previous_error,
                                                                         previous_derivative=previous_derivative,
                                                                         dt=dt)
    return throttle, error, new_derivative

def deflection_control(effective_angle_of_attack, previous_error, previous_derivative, dt):
    Kp_deflection = -10
    Kd_deflection = 0.01
    N_deflection = 10
    deflection, new_derivative = PD_controller_single_step(Kp=Kp_deflection,
                                                           Kd=Kd_deflection,
                                                           N=N_deflection,
                                                           error=effective_angle_of_attack,
                                                           previous_error=previous_error,
                                                           previous_derivative=previous_derivative,
                                                           dt=dt)
    return deflection, new_derivative

def all_controllers(v_y, v_ref, previous_v_error, previous_v_derivative,
                    effective_angle_of_attack, previous_alpha_effective, previous_alpha_eff_derivative,
                    dt, max_deflection_angle_deg):
    non_nom_throttle, v_error, throttle_control_derivative = throttle_controllers(v_y, v_ref, previous_v_error, previous_v_derivative, dt)
    deflection, deflection_derivative = deflection_control(effective_angle_of_attack, previous_alpha_effective, previous_alpha_eff_derivative, dt)

    u0 = 0 # i.e. no gimballing
    u1 = np.clip(2 * non_nom_throttle - 1, -1, 1)
    u2 = np.clip(-deflection/max_deflection_angle_deg, -1, 1)
    u3 = np.clip(deflection/max_deflection_angle_deg, -1, 1)
    actions = (u0, u1, u2, u3)
    return (non_nom_throttle, deflection,
            v_error, throttle_control_derivative, deflection_derivative,
            actions)

class LandingBurn:
    def __init__(self):
        self.dt = 0.01
        # Read reference initial guess trajectory
        df_reference = pd.read_csv('data/reference_trajectory/landing_burn_optimal/initial_guessreference_profile.csv')
        # interpolate reference, y to v
        self.v_opt_fcn = scipy.interpolate.interp1d(df_reference['altitude'], df_reference['velocity'], kind='cubic', fill_value='extrapolate')

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'landing_burn')
        
        self.max_deflection_angle_deg = 60
        self.controller_func = lambda v_y, v_ref, previous_v_error, previous_v_derivative, \
                    effective_angle_of_attack, previous_alpha_effective, previous_alpha_eff_derivative : \
                        all_controllers(v_y, v_ref, previous_v_error, previous_v_derivative,\
                            effective_angle_of_attack, previous_alpha_effective, previous_alpha_eff_derivative,\
                            self.dt, self.max_deflection_angle_deg)
        
        self.initial_conditions()
        self.initialise_logging()

    def reset(self):
        self.initial_conditions()
        self.initialise_logging()

    def initial_conditions(self):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.previous_v_error = self.v_opt_fcn(self.y) - self.vy
        self.previous_v_derivative = 0.0
        self.alpha_effective = self.gamma - self.theta - math.pi
        self.previous_alpha_effective = self.alpha_effective
        self.previous_alpha_eff_derivative = 0.0

        self.gimbal_angle_deg_prev = 0.0
        self.delta_command_left_rad_prev = 0.0
        self.delta_command_right_rad_prev = 0.0
        self.wind_generator = None
    
    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.theta_vals = []
        self.theta_dot_vals = []
        self.gamma_vals = []
        self.alpha_vals = []
        self.mass_vals = []
        self.m_prop_vals = []
        self.time_vals = []

        self.u0_vals = []
        self.u1_vals = []
        self.u2_vals = []
        self.u3_vals = []

        self.vy_ref_vals = []
        self.delta_L_vals = []
        self.delta_R_vals = []
        self.throttle_vals = []

        self.alpha_effective_vals = []

    def closed_loop_step(self):
        v_ref = self.v_opt_fcn(self.y)
        non_nom_throttle, deflection, self.previous_v_error, self.previous_v_derivative,\
            self.previous_alpha_eff_derivative, actions = self.controller_func(self.vy, v_ref, self.previous_v_error, self.previous_v_derivative,
                                                                  self.alpha_effective, self.previous_alpha_effective, self.previous_alpha_eff_derivative)
        self.state, info = self.simulation_step_lambda(self.state, actions,
                                                self.gimbal_angle_deg_prev, self.delta_command_left_rad_prev,
                                                self.delta_command_right_rad_prev, self.wind_generator)
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.previous_alpha_effective = self.alpha_effective
        self.alpha_effective = self.gamma - self.theta - math.pi

        self.gimbal_angle_deg_prev = info['action_info']['gimbal_angle_deg']
        self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
        self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']

        self.x_vals.append(self.x)
        self.y_vals.append(self.y)
        self.vx_vals.append(self.vx)
        self.vy_vals.append(self.vy)
        self.theta_vals.append(self.theta)
        self.theta_dot_vals.append(self.theta_dot)
        self.gamma_vals.append(self.gamma)
        self.alpha_vals.append(self.alpha)
        self.mass_vals.append(self.mass)
        self.m_prop_vals.append(self.mass_propellant)
        self.time_vals.append(self.time)
        self.u0_vals.append(actions[0])
        self.u1_vals.append(actions[1])
        self.u2_vals.append(actions[2])
        self.u3_vals.append(actions[3])

        self.vy_ref_vals.append(v_ref)
        self.delta_L_vals.append(self.delta_command_left_rad_prev)
        self.delta_R_vals.append(self.delta_command_right_rad_prev)
        self.throttle_vals.append(info['action_info']['throttle'])

        self.alpha_effective_vals.append(self.alpha_effective)

    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/landing_burn_optimal/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_landing_burn_control.csv')

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
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': self.theta_vals,
            'theta_dot[rad/s]': self.theta_dot_vals,
            'alpha[rad]': self.alpha_vals,
            'gamma[rad]': self.gamma_vals,
            'mass[kg]': self.mass_vals,
            'masspropellant[kg]': self.m_prop_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals,
            'u2': self.u2_vals,
            'u3': self.u3_vals
        }

        state_action_path = os.path.join(save_folder, 'state_action_landing_burn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)
    
    def plot_results(self):
        
        plt.figure(figsize=(20,15))
        plt.suptitle('Landing Burn (Initial Guess) Control', fontsize=24)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.x_vals, self.y_vals, linewidth=4, color = 'blue')
        ax1.set_xlabel(r'x [$m$]', fontsize=20)
        ax1.set_ylabel(r'y [$m$]', fontsize=20)
        ax1.set_title('Trajectory', fontsize=22)
        ax1.grid(True)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, self.m_prop_vals, linewidth=4, color = 'blue')
        ax2.set_xlabel(r'Time [$s$]', fontsize=20)
        ax2.set_ylabel(r'Mass [$kg$]', fontsize=20)
        ax2.set_title('Propellant mass', fontsize=22)
        ax2.grid(True)
        ax2.tick_params(labelsize=16)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, np.abs(np.array(self.vy_vals)), linewidth=4, color = 'blue', label='Actual')
        ax3.plot(self.time_vals, self.vy_ref_vals, linewidth=2, color = 'red', linestyle='--', label='Reference')
        ax3.set_xlabel(r'Time [$s$]', fontsize=20)
        ax3.set_ylabel(r'Velocity (Absolute) [$m/s$]', fontsize=20)
        ax3.set_title('Velocity', fontsize=22)
        ax3.grid(True)
        ax3.tick_params(labelsize=16)
        ax3.legend(fontsize=20)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.throttle_vals, linewidth=4, color = 'blue')
        ax4.set_xlabel(r'Time [$s$]', fontsize=20)
        ax4.set_ylabel(r'Throttle', fontsize=20)
        ax4.set_title('Throttle', fontsize=22)
        ax4.grid(True)
        ax4.tick_params(labelsize=16)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, np.rad2deg(np.array(self.alpha_effective_vals)), linewidth=4, color = 'blue')
        ax5.set_xlabel(r'Time [$s$]', fontsize=20)
        ax5.set_ylabel(r'Effective angle of attack [$^{\circ}$]', fontsize=20)
        ax5.set_title('Effective angle of attack', fontsize=22)
        ax5.grid(True)
        ax5.tick_params(labelsize=16)

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, np.rad2deg(np.array(self.delta_L_vals)), linewidth=4, color = 'magenta', label='Left')
        ax6.plot(self.time_vals, np.rad2deg(np.array(self.delta_R_vals)), linewidth=4, color = 'cyan', label='Right')
        ax6.set_xlabel(r'Time [$s$]', fontsize=20)
        ax6.set_ylabel(r'Deflection angle [$^{\circ}$]', fontsize=20)
        ax6.set_title('Deflection angle', fontsize=22)
        ax6.grid(True)
        ax6.tick_params(labelsize=16)
        ax6.legend(fontsize=20)

        plt.savefig('results/landing_burn_optimal/landing_burn_control_initial_guess.png')
        plt.close()
        
    def run_closed_loop(self):
        while self.mass_propellant > 0 and self.y > 1:
            self.closed_loop_step()
        self.save_results()
        self.plot_results()