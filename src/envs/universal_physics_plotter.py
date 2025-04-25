import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
from src.TrajectoryGeneration.Transformations import calculate_flight_path_angles

def universal_physics_plotter(env,
                              agent,
                              save_path,
                              flight_phase = None,
                              type = 'pso'):
    assert type in ['pso', 'rl', 'physics', 'supervisory']
    assert env.flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []
    theta_array = []
    theta_dot_array = []
    gamma_array = []
    alpha_array = []
    mass_array = []
    acceleration_x_component_control = []
    acceleration_y_component_control = []
    acceleration_x_component_drag = []
    acceleration_y_component_drag = []
    acceleration_x_component_gravity = []
    acceleration_y_component_gravity = []
    acceleration_x_component_lift = []
    acceleration_y_component_lift = []
    acceleration_x_component = []
    acceleration_y_component = []
    mass_propellant_array = []


    dynamic_pressures = []
    mach_numbers = []
    mach_numbers_max = []
    CLs = []
    CDs = []

    moments = []
    control_moment = []
    moments_aero = []
    inertia = []

    control_force_parallel = []
    control_force_perpendicular = []

    d_cp_cg = []
    d_thrust_cg = []

    gimbal_angle_deg = []
    throttle = []
    u0 = []
    u1 = []
    u2 = []
    grid_fin_deflection_left_deg = []
    grid_fin_deflection_right_deg = []
    RCS_throttles = []

    effective_angles_of_attack = []

    time = []

    
    done_or_truncated = False
    state = env.reset()
    while not done_or_truncated:
        if type == 'pso':
            actions = agent.forward(state)
            state, reward, done, truncated, info = env.step(actions)
            done_or_truncated = done or truncated
        elif type == 'rl':
            actions = agent.select_actions_no_stochastic(state)
            state, reward, done, truncated, info = env.step(actions)
            done_or_truncated = done or truncated
        elif type == 'physics':
            time_to_break = 300
            target_altitude = 70000
            actions = np.array([0, 0, 0])
            state, terminated, info = env.physics_step_test(actions, target_altitude)
            done_or_truncated = terminated or state[-1] > time_to_break
        elif type == 'supervisory':
            actions = agent.select_actions_no_stochastic(state)
            state, reward, done, truncated, info = env.step(actions)
            done_or_truncated = done or truncated
        

        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, t = info['state']
        
        x_array.append(x)
        y_array.append(y)
        vx_array.append(vx)
        vy_array.append(vy)
        theta_array.append(theta)
        theta_dot_array.append(theta_dot)
        gamma_array.append(gamma)
        alpha_array.append(alpha)
        mass_array.append(mass)
        mass_propellant_array.append(mass_propellant)
        time.append(t)
        effective_angles_of_attack.append(gamma - theta - math.pi)

        acceleration_dict = info['acceleration_dict']
        acceleration_x_component_control.append(acceleration_dict['acceleration_x_component_control'])
        acceleration_y_component_control.append(acceleration_dict['acceleration_y_component_control'])
        acceleration_x_component_drag.append(acceleration_dict['acceleration_x_component_drag'])
        acceleration_y_component_drag.append(acceleration_dict['acceleration_y_component_drag'])
        acceleration_x_component_gravity.append(acceleration_dict['acceleration_x_component_gravity'])
        acceleration_y_component_gravity.append(acceleration_dict['acceleration_y_component_gravity'])
        acceleration_x_component_lift.append(acceleration_dict['acceleration_x_component_lift'])
        acceleration_y_component_lift.append(acceleration_dict['acceleration_y_component_lift'])
        acceleration_x_component.append(acceleration_dict['acceleration_x_component'])
        acceleration_y_component.append(acceleration_dict['acceleration_y_component'])
        
        mach_numbers.append(info['mach_number'])
        mach_numbers_max.append(info['mach_number_max'])
        dynamic_pressures.append(info['dynamic_pressure'])
        CLs.append(info['CL'])
        CDs.append(info['CD'])
        moments.append(info['moment_dict']['moments_z'])
        control_moment.append(info['moment_dict']['control_moment_z'])
        moments_aero.append(info['moment_dict']['aero_moment_z'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])
        if env.flight_phase in ['subsonic', 'supersonic']:
            gimbal_angle_deg.append(info['action_info']['gimbal_angle_deg'])
            throttle.append(info['action_info']['throttle'])
        elif env.flight_phase == 'flip_over_boostbackburn':
            gimbal_angle_deg.append(info['action_info']['gimbal_angle_deg'])
            throttle.append(1.0)
            u0.append(actions[0])
        elif env.flight_phase == 'ballistic_arc_descent':
            u0.append(actions[0])
            RCS_throttles.append(info['action_info']['RCS_throttle'])
        elif env.flight_phase == 're_entry_burn':
            u0.append(actions[0])
            u1.append(actions[1])
            u2.append(actions[2])
            throttle.append(info['action_info']['throttle'])
            grid_fin_deflection_left_deg.append(info['action_info']['delta_left_deg'])
            grid_fin_deflection_right_deg.append(info['action_info']['delta_right_deg'])

        control_force_parallel.append(info['control_force_parallel'])
        control_force_perpendicular.append(info['control_force_perpendicular'])

    if type == 'pso' or type == 'rl':
        print(f'Mach number: {max(mach_numbers)}')
        truncation_id = env.truncation_id()
        if env.flight_phase in ['subsonic', 'supersonic']:
            if truncation_id == 0:
                print(f'It is done, Jonny go have a cerveza.')
            elif truncation_id == 1:
                print(f'Truncated as propellant is depleted.')
            elif truncation_id == 2:
                print(f'Truncated as mach number is too high.')
            elif truncation_id == 3:
                print(f'Truncated as x_error is too high.')
            elif truncation_id == 4:
                print(f'Truncated as negative altitude; should not happen.')
            elif truncation_id == 5:
                print(f'Truncated as alpha is too high.')
            elif truncation_id == 6:
                print(f'Truncated as vx error is too high.')
            elif truncation_id == 7:
                print(f'Truncated as vy error is too high.')
            else:
                print(f'Truncated as unknown reason; truncation_id: {truncation_id}')
        elif env.flight_phase == 'flip_over_boostbackburn':
            if truncation_id == 0:
                print(f'It is done, Jonny go have a cerveza.')
            elif truncation_id == 1:
                print(f'Truncated as propellant is depleted.')
            elif truncation_id == 2:
                print(f'Truncated as pitch error is too high')
            else:
                print(f'Truncated as unknown reason; truncation_id: {truncation_id}')
        elif env.flight_phase == 'ballistic_arc_descent':
            if truncation_id == 0:
                print(f'It is done, Jonny go have a cerveza.')
            elif truncation_id == 1:
                print(f'Effective angle of attack is too high.')
            else:
                print(f'Truncated as unknown reason; truncation_id: {truncation_id}')
        elif env.flight_phase == 're_entry_burn':
            if truncation_id == 0:
                print(f'It is done, Jonny go have a cerveza.')
            elif truncation_id == 1:
                print(f'Truncated as propellant is depleted.')
            elif truncation_id == 2:
                print(f'Truncated as dynamic pressure is too high.')
            elif truncation_id == 3:
                print(f'Truncated as altitude is too low.')
            else:
                print(f'Truncated as unknown reason; truncation_id: {truncation_id}')


    if len(time) > 0:
        plt.rcParams.update({'font.size': 14})
        
        plt.figure(figsize=(40, 40))
        # 5 rows, 4 columns
        gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1], hspace=0.7, wspace=0.5)

        # Subplot 1: x vs Time
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(time, x_array, color='blue', linewidth=2)
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel('Horizontal position [m]', fontsize=20)
        ax1.set_title('Horizontal position', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True)

        # Subplot 2: y vs Time
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(time, np.array(y_array), color='green', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_ylabel('Altitude [m]', fontsize=20)
        ax2.set_title('Altitude', fontsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.grid(True)

        # Subplot 3: vx vs Time
        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(time, vx_array, color='red', linewidth=2)
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.set_ylabel('Horizontal velocity [m/s]', fontsize=20)
        ax3.set_title('Horizontal velocity', fontsize=22)
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.grid(True)

        # Subplot 4: vy vs Time
        ax4 = plt.subplot(gs[0, 3])
        ax4.plot(time, np.array(vy_array), color='purple', linewidth=2)
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.set_ylabel('Vertical velocity [m/s]', fontsize=20)
        ax4.set_title('Vertical velocity', fontsize=22)
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 0])
        ax5.plot(time, np.rad2deg(theta_array), label='Pitch', color='orange', linewidth=2)
        ax5.plot(time, np.rad2deg(gamma_array), label='Flight path', color='cyan', linewidth=2)
        ax5.set_xlabel('Time [s]', fontsize=20)
        ax5.set_ylabel('Angle [deg]', fontsize=20)
        ax5.set_title('Euler Angles', fontsize=22)
        ax5.tick_params(axis='both', which='major', labelsize=18)
        ax5.legend(fontsize=18)
        ax5.grid(True)

        ax6 = plt.subplot(gs[1, 1])
        ax6.plot(time, np.rad2deg(theta_dot_array), color='brown', linewidth=2)
        ax6.set_xlabel('Time [s]', fontsize=20)
        ax6.set_ylabel('Pitch rate [deg/s]', fontsize=20)
        ax6.set_title('Pitch Rate', fontsize=22)
        ax6.tick_params(axis='both', which='major', labelsize=18)
        ax6.grid(True)

        ax7 = plt.subplot(gs[1, 2])
        ax7.plot(time, np.array(mass_array)/1000, color='black', label='Mass', linewidth=2)
        if env.flight_phase in ['flip_over_boostbackburn', 're_entry_burn']:
            ax7.plot(time, np.array(mass_propellant_array)/1000, color='red', label='Mass Propellant', linewidth=2)
        ax7.set_xlabel('Time [s]', fontsize=20)
        ax7.set_ylabel('Mass [ton]', fontsize=20)
        ax7.set_title('Mass', fontsize=22)
        if env.flight_phase in ['flip_over_boostbackburn', 're_entry_burn']:
            ax7.legend(fontsize=18)
        ax7.grid(True)

        ax8 = plt.subplot(gs[1, 3])
        ax8.plot(time, np.array(mach_numbers), color='black', label='Mach Number', linewidth=2)
        if max(mach_numbers) > 0.8:
            ax8.axhline(y=0.8, color='r', linestyle='--')
        if max(mach_numbers) > 1.2:
            ax8.axhline(y=1.2, color='r', linestyle='--')
        ax8.set_xlabel('Time [s]', fontsize=20)
        ax8.set_ylabel('Mach [-]', fontsize=20)
        ax8.set_title('Mach Number', fontsize=22)
        ax8.tick_params(axis='both', which='major', labelsize=18)
        ax8.grid(True)

        ax9 = plt.subplot(gs[2, 0])
        ax9.plot(time, np.array(acceleration_x_component), color='black', label='Total', linestyle='--', linewidth=2)
        ax9.plot(time, np.array(acceleration_x_component_control), color='red', label='Control', linestyle='-.', linewidth=2)
        ax9.plot(time, np.array(acceleration_x_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax9.set_xlabel('Time [s]', fontsize=20)
        ax9.set_ylabel('Horizontal acceleration [m/s^2]', fontsize=20)
        ax9.set_title('Horizontal Acceleration', fontsize=22)
        ax9.legend(fontsize=18)
        ax9.grid(True)

        ax10 = plt.subplot(gs[2, 1])
        ax10.plot(time, np.array(acceleration_y_component), color='black', label='Total', linewidth=2)
        ax10.plot(time, np.array(acceleration_y_component_control), color='red', label='Control', linewidth=2)
        ax10.plot(time, np.array(acceleration_y_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax10.plot(time, np.array(acceleration_y_component_gravity), color='green', label='Gravity')
        ax10.plot(time, np.array(acceleration_y_component_lift), color='purple', label='Lift', linewidth=1.5)
        ax10.set_xlabel('Time [s]', fontsize=20)
        ax10.set_ylabel('Vertical acceleration [m/s^2]', fontsize=20)
        ax10.set_title('Vertical Acceleration', fontsize=22)
        ax10.legend(fontsize=18)
        ax10.grid(True)

        ax11 = plt.subplot(gs[2, 2])
        ax11.plot(time, np.array(CLs), color='black', label='CL', linewidth=2)
        ax11.set_xlabel('Time [s]', fontsize=20)
        ax11.set_ylabel('Lift Coefficient [-]', fontsize=20)
        ax11.set_title('Lift Coefficient', fontsize=22)
        ax11.tick_params(axis='both', which='major', labelsize=18)
        ax11.grid(True)

        ax12 = plt.subplot(gs[2, 3])
        ax12.plot(time, np.array(CDs), color='black', label='CD', linewidth=2)
        ax12.set_xlabel('Time [s]', fontsize=20)
        ax12.set_ylabel('Drag Coefficient [-]', fontsize=20)
        ax12.set_title('Drag Coefficient', fontsize=22)
        ax12.tick_params(axis='both', which='major', labelsize=18)
        ax12.grid(True)

        ax13 = plt.subplot(gs[3, 0])
        ax13.plot(time, np.array(moments), color='black', label='Total', linewidth=2)
        ax13.plot(time, np.array(control_moment), color='red', label='Control', linewidth=2)
        ax13.plot(time, np.array(moments_aero), color='blue', label='Aero', linewidth=1.5)
        ax13.set_xlabel('Time [s]', fontsize=20)
        ax13.set_ylabel('Moments [Nm]', fontsize=20)
        ax13.set_title('Moments', fontsize=22)
        ax13.legend(fontsize=18)
        ax13.grid(True)

        ax14 = plt.subplot(gs[3, 1])
        ax14.plot(time, np.array(dynamic_pressures)/1000, color='black', label='Dynamic Pressure', linewidth=2)
        ax14.set_xlabel('Time [s]', fontsize=20)
        ax14.set_ylabel('Dynamic pressure [kPa]', fontsize=20)
        ax14.set_title('Dynamic Pressure', fontsize=22)
        ax14.tick_params(axis='both', which='major', labelsize=18)
        ax14.grid(True)

        ax15 = plt.subplot(gs[3, 2])
        ax15.plot(time, np.array(control_force_parallel), color='black', label='Control force parallel', linewidth=2)
        ax15.set_xlabel('Time [s]', fontsize=20)
        ax15.set_ylabel('Parallel thrust [N]', fontsize=20)
        ax15.set_title('Parallel Thrust', fontsize=22)
        ax15.tick_params(axis='both', which='major', labelsize=18)
        ax15.grid(True)

        ax16 = plt.subplot(gs[3, 3])   
        ax16.plot(time, np.array(control_force_perpendicular), color='red', label='Control force perpendicular', linewidth=2)
        ax16.set_xlabel('Time [s]', fontsize=20)
        ax16.set_ylabel('Perpendicular thrust [N]', fontsize=20)
        ax16.set_title('Perpendicular Thrust', fontsize=22)
        ax16.tick_params(axis='both', which='major', labelsize=18)
        ax16.grid(True)

        ax17 = plt.subplot(gs[4, 0])
        if env.flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']:
            ax17.plot(time, np.array(gimbal_angle_deg), color='black', label='Gimbal Angle', linewidth=2)
            ax17.set_xlabel('Time [s]', fontsize=20)
            ax17.set_ylabel('Gimbal angle [deg]', fontsize=20)
            ax17.set_title('Gimbal Angle', fontsize=22)
        elif env.flight_phase == 'ballistic_arc_descent':
            # leave empty
            ax17.plot(time, np.array(RCS_throttles), color='black', label='RCS Throttle', linewidth=2)
            ax17.set_xlabel('Time [s]', fontsize=20)
            ax17.set_ylabel('RCS throttle [-]', fontsize=20)
            ax17.set_title('RCS Throttle', fontsize=22)
        elif env.flight_phase == 're_entry_burn':
            ax17.plot(time, np.array(grid_fin_deflection_left_deg), color='black', label='Left', linewidth=2)
            ax17.plot(time, np.array(grid_fin_deflection_right_deg), color='red', linestyle='--', label='Right', linewidth=2)
            ax17.set_xlabel('Time [s]', fontsize=20)
            ax17.set_ylabel('Deflection [deg]', fontsize=20)
            ax17.set_title('Grid Fin Deflection', fontsize=22)
            ax17.legend(fontsize=18)
        ax17.grid(True)

        ax18 = plt.subplot(gs[4, 1])
        if env.flight_phase in ['subsonic', 'supersonic']:  
            ax18.plot(time, np.array(throttle), color='black', label='Throttle', linewidth=2)
            ax18.set_xlabel('Time [s]', fontsize=20)
            ax18.set_ylabel('Throttle [-]', fontsize=20)
            ax18.set_title('Main Engine Throttle', fontsize=22)
        elif env.flight_phase in ['flip_over_boostbackburn', 'ballistic_arc_descent']:
            ax18.plot(time, np.ones_like(time), color='black', label='u0', linewidth=2)
            ax18.set_xlabel('Time [s]', fontsize=20)
            ax18.set_ylabel('Throttle [-]', fontsize=20)
            ax18.set_title('Main Engine Throttle', fontsize=22)
        elif env.flight_phase == 're_entry_burn':
            ax18.plot(time, np.array(RCS_throttles), color='black', label='RCS Throttle', linewidth=2)
            ax18.set_xlabel('Time [s]', fontsize=20)
            ax18.set_ylabel('RCS throttle [-]', fontsize=20)
            ax18.set_title('RCS Throttle', fontsize=22)
        ax18.grid(True)

        ax19 = plt.subplot(gs[4, 2])
        ax19.plot(time, np.array(inertia), color='black', label='Inertia', linewidth=2)
        ax19.set_xlabel('Time [s]', fontsize=20)
        ax19.set_ylabel('Inertia [kg m^2]', fontsize=20)
        ax19.set_title('Inertia', fontsize=22)
        ax19.grid(True)

        ax20 = plt.subplot(gs[4, 3])
        ax20.set_xlabel('Time [s]', fontsize=20)
        ax20.set_ylabel('Angle of Attack [deg]', fontsize=20)
        if env.flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']:
            ax20.plot(time, np.rad2deg(alpha_array), label='alpha', color='magenta', linewidth=2)
            ax20.set_title('Angle of Attack', fontsize=22)
        elif env.flight_phase in ['ballistic_arc_descent', 're_entry_burn']:
            ax20.plot(time, np.rad2deg(np.array(effective_angles_of_attack)), label='alpha', color='magenta', linewidth=2)
            ax20.set_title('Effective Angle of Attack (down)', fontsize=22)
        ax20.grid(True)

        plt.savefig(save_path + 'Simulation.png')
        plt.close()

        # Reference tracking plot
        if not type == 'physics':
            assert flight_phase is not None
            plt.rcParams.update({'font.size': 14})
        
            reference_trajectory_func, _ = reference_trajectory_lambda_func_y(flight_phase)
            xr_array = []
            yr_array = []
            vxr_array = []
            vyr_array = []
            gamma_r_array = []
            for i, y_val in enumerate(y_array):
                xr, yr, vxr, vyr, _ = reference_trajectory_func(y_val)
                xr_array.append(xr)
                yr_array.append(yr)
                vxr_array.append(vxr)
                vyr_array.append(vyr)
                gamma_r = calculate_flight_path_angles(vyr, vxr) # degrees
                if gamma_r < 0:
                    gamma_r = math.degrees(2 * math.pi) + gamma_r
                gamma_r_array.append(gamma_r)

            alpha_r_array = [0 for _ in range(len(time))]
            alpha_effective_r_array = [0 for _ in range(len(time))]

            plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

            ax1 = plt.subplot(gs[0, 0])
            if env.flight_phase == 'subsonic':
                ax1.plot(time, np.array(x_array), color='blue', label='agent', linewidth=2)
                ax1.plot(time, np.array(xr_array), color='red', label='reference', linestyle='--', linewidth=2)
                ax1.set_ylabel('Horizontal position [m]', fontsize=20)
            elif env.flight_phase in ['supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']:
                ax1.plot(time, np.array(x_array)/1000, color='blue', label='agent', linewidth=2)
                ax1.plot(time, np.array(xr_array)/1000, color='red', label='reference', linestyle='--', linewidth=2)
                ax1.set_ylabel('Horizontal position [km]', fontsize=20)
            ax1.set_xlabel('Time [s]', fontsize=20)
            ax1.set_title('Horizontal position', fontsize=22)
            ax1.legend(fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=18)
            ax1.grid(True)

            ax2 = plt.subplot(gs[0, 1])
            ax2.plot(time, np.array(y_array)/1000, color='green', label='agent', linewidth=2)
            ax2.set_xlabel('Time [s]', fontsize=20)
            ax2.set_ylabel('Altitude [km]', fontsize=20)
            ax2.set_title('Altitude', fontsize=22)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.grid(True)

            ax3 = plt.subplot(gs[1, 0])
            ax3.plot(time, np.array(vx_array), color='blue', label='agent', linewidth=2)
            ax3.plot(time, np.array(vxr_array), color='red', label='reference', linestyle='--', linewidth=2)
            ax3.set_xlabel('Time [s]', fontsize=20)
            ax3.set_ylabel('Horizontal velocity [m/s]', fontsize=20)
            ax3.set_title('Horizontal velocity', fontsize=22)
            ax3.legend(fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=18)
            ax3.grid(True)

            ax4 = plt.subplot(gs[1, 1])
            ax4.plot(time, np.array(vy_array), color='blue', label='agent', linewidth=2)
            ax4.plot(time, np.array(vyr_array), color='red', label='reference', linestyle='--', linewidth=2)
            ax4.set_xlabel('Time [s]', fontsize=20)
            ax4.set_ylabel('Vertical velocity [m/s]', fontsize=20)
            ax4.set_title('Vertical velocity', fontsize=22)
            ax4.legend(fontsize=20)
            ax4.tick_params(axis='both', which='major', labelsize=18)
            ax4.grid(True)

            ax5 = plt.subplot(gs[2, 0])
            ax5.plot(time, np.rad2deg(np.array(gamma_array)), color='blue', label='agent', linewidth=2)
            ax5.plot(time, np.array(gamma_r_array), color='red', label='reference', linestyle='--', linewidth=2)
            ax5.set_xlabel('Time [s]', fontsize=20)
            ax5.set_ylabel('Flight Path Angle [deg]', fontsize=20)
            ax5.set_title('Flight Path Angle', fontsize=22)
            ax5.legend(fontsize=20)
            ax5.tick_params(axis='both', which='major', labelsize=18)   
            ax5.grid(True)

            ax6 = plt.subplot(gs[2, 1])
            ax6.set_xlabel('Time [s]', fontsize=20)
            ax6.set_ylabel('Angle of Attack [deg]', fontsize=20)
            if flight_phase in ['subsonic', 'supersonic']:
                ax6.plot(time, np.rad2deg(np.array(alpha_array)), color='blue', label='agent', linewidth=2)
                ax6.plot(time, np.array(alpha_r_array), color='red', label='reference', linestyle='--', linewidth=2)
                ax6.set_title('Angle of Attack', fontsize=22)
            elif flight_phase == 'flip_over_boostbackburn':
                ax6.plot(time, np.rad2deg(np.array(alpha_array)), color='blue', label='agent', linewidth=2)
            elif flight_phase in ['ballistic_arc_descent', 're_entry_burn']:
                ax6.plot(time, np.rad2deg(np.array(effective_angles_of_attack)), color='blue', label='agent', linewidth=2)
                ax6.plot(time, np.rad2deg(np.array(alpha_effective_r_array)), color='red', label='reference', linestyle='--', linewidth=2)
                ax6.set_title('Effective Alpha (bottom) over Time', fontsize=22)
            
            ax6.legend(fontsize=20)
            ax6.tick_params(axis='both', which='major', labelsize=18)
            ax6.grid(True)
            plt.savefig(save_path + 'ReferenceTracking.png')
            plt.close()

            plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

            ax1 = plt.subplot(gs[0, 0])
            ax1.plot(time, np.rad2deg(np.array(gamma_array)), color='blue', label='Flight path', linewidth=2)
            ax1.plot(time, np.array(gamma_r_array), color='red', label='Reference', linestyle='--', linewidth=2)
            ax1.plot(time, np.rad2deg(theta_array), color='green', label='Pitch', linewidth=2)
            ax1.set_xlabel('Time [s]', fontsize=20)
            ax1.set_ylabel('Angle [deg]', fontsize=20)
            ax1.set_title('Angle Tracking', fontsize=22)
            ax1.legend(fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=18)
            ax1.grid(True)

            ax2 = plt.subplot(gs[0, 1])
            ax2.plot(time, np.rad2deg(theta_dot_array), color='blue', label='Pitch rate', linewidth=2)
            ax2.set_xlabel('Time [s]', fontsize=20)
            ax2.set_ylabel('Pitch rate [deg/s]', fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.grid(True)

            ax3 = plt.subplot(gs[1, 0])
            ax3.plot(time, np.array(moments), color='black', label='Total', linewidth=2)
            ax3.plot(time, np.array(control_moment), color='red', label='Control', linewidth=2)
            ax3.plot(time, np.array(moments_aero), color='blue', label='Aero', linewidth=1.5)
            ax3.set_xlabel('Time [s]', fontsize=20)
            ax3.set_ylabel('Moments [Nm]', fontsize=20)
            ax3.set_title('Moments', fontsize=22)
            ax3.legend(fontsize=18)
            ax3.grid(True)

            ax4 = plt.subplot(gs[1, 1])
            if env.flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']:
                ax4.plot(time, np.array(gimbal_angle_deg), color='black', label='Gimbal Angle', linewidth=2)
                ax4.set_xlabel('Time [s]', fontsize=20)
                ax4.set_ylabel('Gimbal angle [deg]', fontsize=20)
                ax4.set_title('Gimbal Angle', fontsize=22)
            elif env.flight_phase == 'ballistic_arc_descent':
                # leave empty
                ax4.plot(time, np.array(RCS_throttles), color='black', label='RCS Throttle', linewidth=2)
                ax4.set_xlabel('Time [s]', fontsize=20)
                ax4.set_ylabel('RCS throttle [-]', fontsize=20)
                ax4.set_title('RCS Throttle', fontsize=22)
            elif env.flight_phase == 're_entry_burn':
                ax4.plot(time, np.array(grid_fin_deflection_left_deg), color='black', label='Left', linewidth=2)
                ax4.plot(time, np.array(grid_fin_deflection_right_deg), color='red', linestyle='--', label='Right', linewidth=2)
                ax4.set_xlabel('Time [s]', fontsize=20)
                ax4.set_ylabel('Deflection [deg]', fontsize=20)
                ax4.set_title('Grid Fin Deflection', fontsize=22)
                ax4.legend(fontsize=18)
            ax4.grid(True)

            ax5 = plt.subplot(gs[2, 0])
            ax5.plot(time, np.array(CLs), color='blue', label='Lift', linewidth=2)
            ax5.set_xlabel('Time [s]', fontsize=20)
            ax5.set_ylabel('Lift Coefficient [-]', fontsize=20)
            ax5.set_title('Lift Coefficient', fontsize=22)
            ax5.legend(fontsize=18)
            ax5.grid(True)

            ax6 = plt.subplot(gs[2, 1])
            ax6.plot(time, np.array(CDs), color='blue', label='Drag', linewidth=2)
            ax6.set_xlabel('Time [s]', fontsize=20)
            ax6.set_ylabel('Drag Coefficient [-]', fontsize=20)
            ax6.set_title('Drag Coefficient', fontsize=22)
            ax6.legend(fontsize=18)
            ax6.grid(True)

            plt.savefig(save_path + 'AngleTracking.png')
            plt.close() 


            force_y_control = np.array(acceleration_y_component_control) * mass_array
            force_y_drag = np.array(acceleration_y_component_drag) * mass_array
            force_y_gravity = np.array(acceleration_y_component_gravity) * mass_array
            force_y_lift = np.array(acceleration_y_component_lift) * mass_array
            force_y_total = np.array(acceleration_y_component) * mass_array
            force_y_aero = force_y_drag + force_y_lift

            plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

            ax1 = plt.subplot(gs[0, 0])
            ax1.plot(time, np.array(mach_numbers), color='blue', label='Mach number', linewidth=2)
            if env.flight_phase in ['subsonic', 'supersonic', 'ballistic_arc_descent', 're_entry_burn']:
                ax1.plot(time, np.array(mach_numbers_max), color='red', label='Maximum', linestyle='--', linewidth=2)
            ax1.set_xlabel('Time [s]', fontsize=20)
            ax1.set_ylabel('Mach number [-]', fontsize=20)
            ax1.set_title('Mach number', fontsize=22)
            ax1.legend(fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=18)
            ax1.set_ylim(top=min(5, max(mach_numbers_max)))
            ax1.grid(True)

            ax2 = plt.subplot(gs[0, 1])
            if max(dynamic_pressures) > 2000:
                ax2.plot(time, np.array(dynamic_pressures)/1000, color='blue', label='Dynamic pressure', linewidth=2)
                ax2.set_ylabel('Dynamic pressure [kPa]', fontsize=20)
                if max(dynamic_pressures) > 25000:
                    ax2.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Maximum')
            else:
                ax2.plot(time, np.array(dynamic_pressures), color='blue', label='Dynamic pressure', linewidth=2)
                ax2.set_ylabel('Dynamic pressure [Pa]', fontsize=20)
            ax2.set_xlabel('Time [s]', fontsize=20)
            ax2.set_title('Dynamic Pressure', fontsize=22)
            ax2.legend(fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.grid(True)

            ax3 = plt.subplot(gs[1, 0])
            if max(control_force_parallel) > 1e6:
                ax3.plot(time, np.array(control_force_parallel)/1e6, color='blue', linewidth=2)
                ax3.set_ylabel('Control force parallel [MN]', fontsize=20)
            else:
                ax3.plot(time, np.array(control_force_parallel)/1e3, color='blue', linewidth=2)
                ax3.set_ylabel('Force [kN]', fontsize=20)
            ax3.set_xlabel('Time [s]', fontsize=20)
            ax3.set_title('Control Force Parallel', fontsize=22)
            ax3.tick_params(axis='both', which='major', labelsize=18)
            ax3.grid(True)

            ax4 = plt.subplot(gs[1, 1])
            if max(control_force_perpendicular) > 1e6:
                ax4.plot(time, np.array(control_force_perpendicular)/1e6, color='blue', linewidth=2)
                ax4.set_ylabel('Control force perpendicular [MN]', fontsize=20)
            else:
                ax4.plot(time, np.array(control_force_perpendicular)/1e3, color='blue', linewidth=2)
                ax4.set_ylabel('Force [kN]', fontsize=20)
            ax4.set_xlabel('Time [s]', fontsize=20)
            ax4.set_title('Control Force Perpendicular', fontsize=22)
            ax4.tick_params(axis='both', which='major', labelsize=18)
            ax4.grid(True)

            ax5 = plt.subplot(gs[2, 0])
            ax5.plot(time, np.array(mass_propellant_array)/1000, color='blue', linewidth=2)
            ax5.set_xlabel('Time [s]', fontsize=20)
            ax5.set_ylabel('Mass [ton]', fontsize=20)
            ax5.set_title('Propellant Mass', fontsize=22)
            ax5.tick_params(axis='both', which='major', labelsize=18)
            ax5.grid(True)

            ax6 = plt.subplot(gs[2, 1])
            if max(force_y_total) > 1e6:
                ax6.plot(time, np.array(force_y_total)/1e6, color='black', label='Total', linewidth=2)
                ax6.plot(time, np.array(force_y_control)/1e6, color='purple', label='Control', linewidth=2)
                ax6.plot(time, np.array(force_y_aero)/1e6, color='orange', label='Aerodynamic', linewidth=2)
                ax6.plot(time, np.array(force_y_gravity)/1e6, color='green', label='Gravity')
                ax6.set_ylabel('Force [MN]', fontsize=20)
            else:
                ax6.plot(time, np.array(force_y_total)/1e3, color='black', label='Total', linewidth=2)
                ax6.plot(time, np.array(force_y_control)/1e3, color='purple', label='Control', linewidth=2)
                ax6.plot(time, np.array(force_y_aero)/1e3, color='orange', label='Aerodynamic', linewidth=2)
                ax6.plot(time, np.array(force_y_gravity)/1e3, color='green', label='Gravity')
                ax6.set_ylabel('Force [kN]', fontsize=20)
            ax6.set_xlabel('Time [s]', fontsize=20)
            ax6.set_title('Vertical Force', fontsize=22)
            ax6.legend(fontsize=20)
            ax6.tick_params(axis='both', which='major', labelsize=18)
            ax6.grid(True)

            plt.savefig(save_path + 'VerticalMotion.png')
            plt.close()

        # Now an x - y plot with the same scalled axis
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.plot(x_array, y_array, color='blue', label='Trajectory', linewidth=2)
        ax.scatter(x_array[0], y_array[0], color='green', label='Start', s=100, zorder=5)
        ax.scatter(x_array[-1], y_array[-1], color='red', label='End', s=100, zorder=5)
        ax.set_xlabel('Horizontal position [m]', fontsize=20)
        ax.set_ylabel('Altitude [m]', fontsize=20)
        ax.set_title('X - Y trajectory', fontsize=22)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(True)
        plt.savefig(save_path + 'XY_Trajectory.png')
        plt.close()

        if type in ['pso', 'rl', 'supervisory']:
            model_name = save_path.split('/')[-2]
            if type == 'pso':
                data_save_path = f'data/pso_saves/{model_name}/'
            elif type == 'rl':
                data_save_path = f'data/agent_saves/VanillaSAC/{model_name}/'
            elif type == 'supervisory':
                data_save_path = f'data/agent_saves/SupervisoryLearning/{model_name}/'
            # Save data to csv
            data = {
                'time[s]': time,
                'x[m]': x_array,
                'y[m]': y_array,
                'vx[m/s]': vx_array,
                'vy[m/s]': vy_array,
                'theta[rad]': theta_array,
                'theta_dot[rad/s]': theta_dot_array,
                'gamma[rad]': gamma_array,
                'alpha[rad]': alpha_array,
                'mass[kg]': mass_array,
                'mass_propellant[kg]': mass_propellant_array    
            }
            df = pd.DataFrame(data)
            df.to_csv(data_save_path + 'trajectory.csv', index=False)
    else:
        print("Warning: No simulation data collected. The simulation may have terminated immediately.")