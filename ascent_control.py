import math
import numpy as np
import csv
import matplotlib.pyplot as plt
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
    mach_number_max = math.sqrt(2 * Q_max / air_density) * 1 / speed_of_sound
    mach_reference_rl = 0.2
    mach_number_reference = max(mach_number_reference_previous - mach_reference_rl, min(mach_number_reference_previous + mach_reference_rl, mach_number_max))
    error_mach_number = mach_number_reference - mach_number
    throttle_non_nom = np.clip(Kp_mach * error_mach_number, -1, 1)

    return throttle_non_nom, mach_number_reference

def gimbal_determination(Mz,
                         non_nominal_throttle,
                         atmospheric_pressure,
                         d_thrust_cg,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area):
    nominal_throttle = 0.5
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

def augment_actions_ascent_control(gimbal_angle_rad, non_nominal_throttle):
    max_gimbal_angle_rad = math.radians(5)

    u0 = gimbal_angle_rad / max_gimbal_angle_rad
    u1 = 2 * non_nominal_throttle - 1

    actions = (u0, u1)
    return actions

class AscentControl:
    def __init__(self):
        self.T_final = 100
        self.dt = 0.1

        self.pitch_reference_lambda = lambda time : ascent_reference_pitch(time, self.T_final)

        self.state = load_subsonic_initial_state()
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'subsonic',
                    initial_state = self.state)
        
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        
        self.gimbal_determiner = lambda Mz, non_nominal_throttle, atmospheric_pressure, d_thrust_cg : gimbal_determination(
            Mz, non_nominal_throttle, atmospheric_pressure, d_thrust_cg,
            number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1']),
            thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1']),
            nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1']),
            nozzle_exit_area = float(sizing_results['Nozzle exit area'])
        )

        self.mach_number_reference_previous = 0.0
        self.initial_conditions()
        self.initialise_logging()

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

    def initial_conditions(self):
        _, info_IC = self.simulation_step_lambda(self.state, (0,0))
        self.atmospheric_pressure = info_IC['atmospheric_pressure']
        self.air_density = info_IC['air_density']
        self.speed_of_sound = info_IC['speed_of_sound']
        self.mach_number = 0.0
        self.d_thrust_cg = info_IC['d_thrust_cg']
        self.time = 0.0
        self.pitch_angle_rad = math.pi/2

    def closed_loop_step(self):
        pitch_reference_rad = self.pitch_reference_lambda(self.time)
        control_moments =  ascent_pitch_controller(pitch_reference_rad, self.pitch_angle_rad)
        non_nominal_throttle, self.mach_number_reference_previous = ascent_controller_step(self.mach_number_reference_previous,
                           self.mach_number,
                           self.air_density,
                           self.speed_of_sound)
        
        gimbal_angle_rad = self.gimbal_determiner(control_moments, non_nominal_throttle, self.atmospheric_pressure, self.d_thrust_cg)        
        actions = augment_actions_ascent_control(gimbal_angle_rad, non_nominal_throttle)

        self.state, info = self.simulation_step_lambda(self.state, actions)
        
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
        plt.plot(self.time_vals, self.mach_number_vals, linewidth = 2, label = 'Mach Number')
        plt.plot(self.time_vals, self.mach_number_reference_vals, linewidth = 2, label = 'Mach Number Reference')
        plt.xlabel('Time [s]')
        plt.ylabel('Mach Number')
        plt.title('Mach Number')
        plt.ylim(0, 4.0)
        plt.grid()
        plt.legend()

        plt.subplot(4, 2, 3)
        plt.plot(self.time_vals, self.pitch_angle_deg_vals, linewidth = 2, label = 'Pitch Angle')
        plt.plot(self.time_vals, self.pitch_angle_reference_deg_vals, linewidth = 2, label = 'Pitch Angle Reference')
        plt.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 2, label = 'Flight Path Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Pitch and Flight Path Angles')
        plt.grid()
        plt.legend()

        plt.subplot(4, 2, 4)
        plt.plot(self.time_vals, self.angle_of_attack_deg_vals, linewidth = 2, label = 'Angle of Attack')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Angle of Attack')
        plt.grid()

        plt.subplot(4, 2, 5)
        plt.plot(self.time_vals, self.gimbal_angle_deg_vals, linewidth = 2, label = 'Gimbal Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Gimbal Angle')
        plt.grid()

        plt.subplot(4, 2, 6)
        plt.plot(self.time_vals, self.non_nominal_throttle_vals, linewidth = 2, label = 'Non Nominal Throttle')
        plt.xlabel('Time [s]')
        plt.ylabel('Throttle')
        plt.title('Non Nominal Throttle')
        plt.grid()

        plt.subplot(4, 2, 7)
        plt.plot(self.time_vals, self.u0_vals, linewidth = 2, label = 'u0')
        plt.xlabel('Time [s]')
        plt.ylabel('u0')
        plt.title('Action: normalised gimbal')
        plt.grid()
        plt.legend()

        plt.subplot(4, 2, 8)
        plt.plot(self.time_vals, self.u1_vals, linewidth = 2, label = 'u1')
        plt.xlabel('Time [s]')
        plt.ylabel('u1')
        plt.title('Action: non-nominal throttle')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'results/ascent_controls/simulation.png')
        plt.close()
        
    def run_closed_loop(self):
        while self.state[-1] < self.T_final:
            self.closed_loop_step()
        self.plot_results()

if __name__ == "__main__":
    ascent_control = AscentControl()
    ascent_control.run_closed_loop()