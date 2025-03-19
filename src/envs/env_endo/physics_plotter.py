import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.env_endo.init_vertical_rising import reference_trajectory_lambda

def test_physics_endo_with_plot(rocket):
    # Old VRandGT physics test.ipynb
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []
    theta_array = []
    theta_dot_array = []
    gamma_array = []
    alpha_array = []
    mass_array = []

    acceleration_x_component_thrust = []
    acceleration_y_component_thrust = []
    acceleration_x_component_drag = []
    acceleration_y_component_drag = []
    acceleration_x_component_gravity = []
    acceleration_y_component_gravity = []
    acceleration_x_component_lift = []
    acceleration_y_component_lift = []
    acceleration_x_component = []
    acceleration_y_component = []

    mach_number = []
    CLs = []
    CDs = []

    moments = []
    moments_thrust = []
    moments_aero = []
    inertia = []

    d_cp_cg = []
    d_thrust_cg = []

    time = []
    t = 0
    time_to_break = 300 # 128
    target_altitude = 70000                  # [m]
    terminated = False
    while not terminated and t < time_to_break:
        actions = 0

        state, terminated, info = rocket.physics_step_test(actions, target_altitude)
        acceleration_dict = info['acceleration_dict']
        acceleration_x_component_thrust.append(acceleration_dict['acceleration_x_component_thrust'])
        acceleration_y_component_thrust.append(acceleration_dict['acceleration_y_component_thrust'])
        acceleration_x_component_drag.append(acceleration_dict['acceleration_x_component_drag'])
        acceleration_y_component_drag.append(acceleration_dict['acceleration_y_component_drag'])
        acceleration_x_component_gravity.append(acceleration_dict['acceleration_x_component_gravity'])
        acceleration_y_component_gravity.append(acceleration_dict['acceleration_y_component_gravity'])
        acceleration_x_component_lift.append(acceleration_dict['acceleration_x_component_lift'])
        acceleration_y_component_lift.append(acceleration_dict['acceleration_y_component_lift'])
        acceleration_x_component.append(acceleration_dict['acceleration_x_component'])
        acceleration_y_component.append(acceleration_dict['acceleration_y_component'])
        mach_number.append(info['mach_number'])
        CLs.append(info['CL'])
        CDs.append(info['CD'])
        moments.append(info['moment_dict']['moments_z'])
        moments_thrust.append(info['moment_dict']['thrust_moments_y'])
        moments_aero.append(info['moment_dict']['aero_moements_y'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])

        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, t = state
        x_array.append(x)
        y_array.append(y)
        vx_array.append(vx)
        vy_array.append(vy)
        theta_array.append(theta)
        theta_dot_array.append(theta_dot)
        gamma_array.append(gamma)
        alpha_array.append(alpha)
        mass_array.append(mass)
        time.append(t)

    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)

    # Subplot 1: x vs Time
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, x_array, color='blue')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('x [m]')
    ax1.set_title('Position x over Time')
    ax1.grid(True)

    # Subplot 2: y vs Time
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, np.array(y_array), color='green')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Position y over Time')
    ax2.grid(True)

    # Subplot 3: vx vs Time
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(time, vx_array, color='red')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('vx [m/s]')
    ax3.set_title('Velocity vx over Time')
    ax3.grid(True)

    # Subplot 4: vy vs Time
    ax4 = plt.subplot(gs[0, 3])
    ax4.plot(time, np.array(vy_array), color='purple')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('vy [m/s]')
    ax4.set_title('Velocity vy over Time')
    ax4.grid(True)

    ax5 = plt.subplot(gs[1, 0])
    ax5.plot(time, np.rad2deg(theta_array), label='theta', color='orange')
    ax5.plot(time, np.rad2deg(gamma_array), label='gamma', color='cyan')
    ax5.plot(time, np.rad2deg(alpha_array), label='alpha', color='magenta')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Angle [deg]')
    ax5.set_title('Euler Angles over Time')
    ax5.legend()
    ax5.grid(True)


    ax6 = plt.subplot(gs[1, 1])
    ax6.plot(time, np.rad2deg(theta_dot_array), color='brown')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('theta_dot [deg/s]')
    ax6.set_title('Theta Dot over Time')
    ax6.grid(True)

    gamma_dot_array = np.rad2deg(np.gradient(gamma_array, time))
    ax7 = plt.subplot(gs[1, 2])
    ax7.plot(time, gamma_dot_array, color='gray')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('gamma_dot [deg/s]')
    ax7.set_title('Gamma Dot over Time')
    ax7.grid(True)


    ax8 = plt.subplot(gs[1, 3])
    ax8.plot(time, np.array(mass_array)/1000, color='black', label='Mass')
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Mass [ton]')
    ax8.set_title('Mass over Time')
    ax8.legend()
    ax8.grid(True)

    ax9 = plt.subplot(gs[2, 0])
    ax9.plot(time, np.array(mach_number), color='black', label='Mass')
    ax9.axhline(y=0.8, color='r', linestyle='--')
    ax9.axhline(y=1.2, color='r', linestyle='--')
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Mach Number [-]')
    ax9.set_title('Mach number over Time')
    ax9.legend()
    ax9.grid(True)

    ax10 = plt.subplot(gs[2, 1])
    ax10.plot(time, np.array(acceleration_x_component), color='black', label='Total', linestyle='--')
    ax10.plot(time, np.array(acceleration_x_component_thrust), color='red', label='Thrust', linestyle='-.')
    ax10.plot(time, np.array(acceleration_x_component_drag), color='blue', label='Drag', linewidth=1.5)
    #ax10.plot(time, np.array(acceleration_x_component_gravity), color='green', label='Gravity')
    #ax10.plot(time, np.array(acceleration_x_component_lift), color='purple', label='Lift')
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Acceleration x [m/s^2]')
    ax10.set_title('Rightward Thrust over Time')
    ax10.legend()
    ax10.grid(True)

    ax11 = plt.subplot(gs[2, 2])
    ax11.plot(time, np.array(acceleration_y_component), color='black', label='Total')
    ax11.plot(time, np.array(acceleration_y_component_thrust), color='red', label='Thrust')
    ax11.plot(time, np.array(acceleration_y_component_drag), color='blue', label='Drag')
    ax11.plot(time, np.array(acceleration_y_component_gravity), color='green', label='Gravity')
    ax11.plot(time, np.array(acceleration_y_component_lift), color='purple', label='Lift')
    ax11.set_xlabel('Time [s]')
    ax11.set_ylabel('Acceleration y [m/s^2]')
    ax11.set_title('Upward Thrust over Time')
    ax11.legend()
    ax11.grid(True)

    ax12 = plt.subplot(gs[2, 3])
    ax12.plot(time, np.array(CLs), color='black', label='CL')
    ax12.set_xlabel('Time [s]')
    ax12.set_ylabel('CL [-]')
    ax12.set_title('Lift Coefficient over Time')
    ax12.legend()
    ax12.grid(True)

    ax13 = plt.subplot(gs[3, 0])
    ax13.plot(time, np.array(CDs), color='black', label='CD')
    ax13.set_xlabel('Time [s]')
    ax13.set_ylabel('CD [-]')
    ax13.set_title('Drag Coefficient over Time')
    ax13.legend()
    ax13.grid(True)

    ax14 = plt.subplot(gs[3, 1])
    ax14.plot(time, np.array(moments), color='black', label='Total')
    ax14.plot(time, np.array(moments_thrust), color='red', label='Thrust')
    ax14.plot(time, np.array(moments_aero), color='blue', label='Aero')
    ax14.set_xlabel('Time [s]')
    ax14.set_ylabel('Moments [Nm]')
    ax14.set_title('Moments over Time')
    ax14.legend()
    ax14.grid(True)

    ax14 = plt.subplot(gs[3, 2])
    ax14.plot(time, np.array(inertia), color='black', label='Inertia')
    ax14.set_xlabel('Time [s]')
    ax14.set_ylabel('Inertia [kg*m^2]')
    ax14.set_title('Inertia over Time')
    ax14.legend()
    ax14.grid(True)

    ax15 = plt.subplot(gs[3, 3])
    ax15.plot(time, np.array(d_cp_cg), color='black', label='d_cp_cg')
    ax15.plot(time, np.array(d_thrust_cg), color='red', label='d_thrust_cg')
    ax15.set_xlabel('Time [s]')
    ax15.set_ylabel('Distance [m]')
    ax15.set_title('Distances over Time')
    ax15.legend()
    ax15.grid(True)
    plt.savefig('results/EndoPhysicsTest/EndoPhysicsTest.png')
    plt.close()



# test_env_vertical_rising
def test_agent_interaction(env,
                           agent,
                           dt,
                           print_bool,
                           rcs_used = False,
                           acs_used = False):
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []
    theta_array = []
    theta_dot_array = []
    gamma_array = []
    alpha_array = []
    mass_array = []
    acceleration_x_component_thrust = []
    acceleration_y_component_thrust = []
    acceleration_x_component_drag = []
    acceleration_y_component_drag = []
    acceleration_x_component_gravity = []
    acceleration_y_component_gravity = []
    acceleration_x_component_lift = []
    acceleration_y_component_lift = []
    acceleration_x_component = []
    acceleration_y_component = []

    mach_number = []
    CLs = []
    CDs = []

    moments = []
    moments_thrust = []
    moments_aero = []
    inertia = []

    gimbal_angle_deg = []
    throttles = []

    d_cp_cg = []
    d_thrust_cg = []

    time = []
    done_or_truncated = False
    state = env.reset()
    while not done_or_truncated:
        actions = agent.select_actions_no_stochatic(state)
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

        time.append(t)

        acceleration_dict = info['acceleration_dict']
        acceleration_x_component_thrust.append(acceleration_dict['acceleration_x_component_thrust'])
        acceleration_y_component_thrust.append(acceleration_dict['acceleration_y_component_thrust'])
        acceleration_x_component_drag.append(acceleration_dict['acceleration_x_component_drag'])
        acceleration_y_component_drag.append(acceleration_dict['acceleration_y_component_drag'])
        acceleration_x_component_gravity.append(acceleration_dict['acceleration_x_component_gravity'])
        acceleration_y_component_gravity.append(acceleration_dict['acceleration_y_component_gravity'])
        acceleration_x_component_lift.append(acceleration_dict['acceleration_x_component_lift'])
        acceleration_y_component_lift.append(acceleration_dict['acceleration_y_component_lift'])
        acceleration_x_component.append(acceleration_dict['acceleration_x_component'])
        acceleration_y_component.append(acceleration_dict['acceleration_y_component'])
        mach_number.append(info['mach_number'])
        CLs.append(info['CL'])
        CDs.append(info['CD'])
        moments.append(info['moment_dict']['moments_z'])
        moments_thrust.append(info['moment_dict']['thrust_moments_z'])
        moments_aero.append(info['moment_dict']['aero_moements_z'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])


        gimbal_angle_deg.append(info['gimbal_angle_deg'])
        throttles.append(info['throttle'])


    if len(time) > 0:
        plt.rcParams.update({'font.size': 14})
        
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1], hspace=0.6, wspace=0.3)

        # Subplot 1: x vs Time
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(time, x_array, color='blue', linewidth=2)
        ax1.set_xlabel('Time [s]', fontsize=16)
        ax1.set_ylabel('x [m]', fontsize=16)
        ax1.set_title('Position x over Time', fontsize=18)
        ax1.grid(True)

        # Subplot 2: y vs Time
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(time, np.array(y_array), color='green', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=16)
        ax2.set_ylabel('y [m]', fontsize=16)
        ax2.set_title('Position y over Time', fontsize=18)
        ax2.grid(True)

        # Subplot 3: vx vs Time
        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(time, vx_array, color='red', linewidth=2)
        ax3.set_xlabel('Time [s]', fontsize=16)
        ax3.set_ylabel('vx [m/s]', fontsize=16)
        ax3.set_title('Velocity vx over Time', fontsize=18)
        ax3.grid(True)

        # Subplot 4: vy vs Time
        ax4 = plt.subplot(gs[0, 3])
        ax4.plot(time, np.array(vy_array), color='purple', linewidth=2)
        ax4.set_xlabel('Time [s]', fontsize=16)
        ax4.set_ylabel('vy [m/s]', fontsize=16)
        ax4.set_title('Velocity vy over Time', fontsize=18)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 0])
        ax5.plot(time, np.rad2deg(theta_array), label='theta', color='orange', linewidth=2)
        ax5.plot(time, np.rad2deg(gamma_array), label='gamma', color='cyan', linewidth=2)
        ax5.plot(time, np.rad2deg(alpha_array), label='alpha', color='magenta', linewidth=2)
        ax5.set_xlabel('Time [s]', fontsize=16)
        ax5.set_ylabel('Angle [deg]', fontsize=16)
        ax5.set_title('Euler Angles over Time', fontsize=18)
        ax5.legend(fontsize=14)
        ax5.grid(True)


        ax6 = plt.subplot(gs[1, 1])
        ax6.plot(time, np.rad2deg(theta_dot_array), color='brown', linewidth=2)
        ax6.set_xlabel('Time [s]', fontsize=16)
        ax6.set_ylabel('theta_dot [deg/s]', fontsize=16)
        ax6.set_title('Theta Dot over Time', fontsize=18)
        ax6.grid(True)

        # Calculate gamma_dot_array only if we have data
        if len(gamma_array) > 1:  # Need at least 2 points for gradient
            gamma_dot_array = np.rad2deg(np.gradient(gamma_array, time))
            ax7 = plt.subplot(gs[1, 2])
            ax7.plot(time, gamma_dot_array, color='yellow', linewidth=2)
            ax7.set_xlabel('Time [s]', fontsize=16)
            ax7.set_ylabel('gamma_dot [deg/s]', fontsize=16)
            ax7.set_title('Gamma Dot over Time', fontsize=18)
            ax7.grid(True)
        else:
            # Create empty plot if not enough data for gradient
            ax7 = plt.subplot(gs[1, 2])
            ax7.set_xlabel('Time [s]', fontsize=16)
            ax7.set_ylabel('gamma_dot [deg/s]', fontsize=16)
            ax7.set_title('Gamma Dot over Time (insufficient data)', fontsize=18)
            ax7.grid(True)

        ax8 = plt.subplot(gs[1, 3])
        ax8.plot(time, np.array(mass_array)/1000, color='black', label='Mass', linewidth=2)
        ax8.set_xlabel('Time [s]', fontsize=16)
        ax8.set_ylabel('Mass [ton]', fontsize=16)
        ax8.set_title('Mass over Time', fontsize=18)
        ax8.legend(fontsize=14)
        ax8.grid(True)

        ax9 = plt.subplot(gs[2, 0])
        ax9.plot(time, np.array(mach_number), color='black', label='Mach Number', linewidth=2)
        ax9.axhline(y=0.8, color='r', linestyle='--')
        ax9.axhline(y=1.2, color='r', linestyle='--')
        ax9.set_xlabel('Time [s]', fontsize=16)
        ax9.set_ylabel('Mach Number [-]', fontsize=16)
        ax9.set_title('Mach number over Time', fontsize=18)
        ax9.grid(True)

        ax10 = plt.subplot(gs[2, 1])
        ax10.plot(time, np.array(acceleration_x_component), color='black', label='Total', linestyle='--', linewidth=2)
        ax10.plot(time, np.array(acceleration_x_component_thrust), color='red', label='Thrust', linestyle='-.', linewidth=2)
        ax10.plot(time, np.array(acceleration_x_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax10.set_xlabel('Time [s]', fontsize=16)
        ax10.set_ylabel('Acceleration x [m/s^2]', fontsize=16)
        ax10.set_title('Rightward Thrust over Time', fontsize=18)
        ax10.legend(fontsize=14)
        ax10.grid(True)

        ax11 = plt.subplot(gs[2, 2])
        ax11.plot(time, np.array(acceleration_y_component), color='black', label='Total', linewidth=2)
        ax11.plot(time, np.array(acceleration_y_component_thrust), color='red', label='Thrust', linewidth=2)
        ax11.plot(time, np.array(acceleration_y_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax11.plot(time, np.array(acceleration_y_component_gravity), color='green', label='Gravity')
        ax11.plot(time, np.array(acceleration_y_component_lift), color='purple', label='Lift', linewidth=1.5)
        ax11.set_xlabel('Time [s]', fontsize=16)
        ax11.set_ylabel('Acceleration y [m/s^2]', fontsize=16)
        ax11.set_title('Upward Thrust over Time', fontsize=18)
        ax11.legend(fontsize=14)
        ax11.grid(True)

        ax12 = plt.subplot(gs[2, 3])
        ax12.plot(time, np.array(CLs), color='black', label='CL', linewidth=2)
        ax12.set_xlabel('Time [s]', fontsize=16)
        ax12.set_ylabel('CL [-]', fontsize=16)
        ax12.set_title('Lift Coefficient over Time', fontsize=18)
        ax12.grid(True)

        ax13 = plt.subplot(gs[3, 0])
        ax13.plot(time, np.array(CDs), color='black', label='CD', linewidth=2)
        ax13.set_xlabel('Time [s]', fontsize=16)
        ax13.set_ylabel('CD [-]', fontsize=16)
        ax13.set_title('Drag Coefficient over Time', fontsize=18)
        ax13.grid(True)

        ax14 = plt.subplot(gs[3, 1])
        ax14.plot(time, np.array(moments), color='black', label='Total', linewidth=2)
        ax14.plot(time, np.array(moments_thrust), color='red', label='Thrust', linewidth=2)
        ax14.plot(time, np.array(moments_aero), color='blue', label='Aero', linewidth=1.5)
        ax14.set_xlabel('Time [s]', fontsize=16)
        ax14.set_ylabel('Moments [Nm]', fontsize=16)
        ax14.set_title('Moments over Time', fontsize=18)
        ax14.legend(fontsize=14)
        ax14.grid(True)

        ax15 = plt.subplot(gs[3, 2])
        ax15.plot(time, np.array(inertia), color='black', label='Inertia', linewidth=2)
        ax15.set_xlabel('Time [s]', fontsize=16)
        ax15.set_ylabel('Inertia [kg*m^2]', fontsize=16)
        ax15.set_title('Inertia over Time', fontsize=18)
        ax15.grid(True)

        ax16 = plt.subplot(gs[3, 3])
        ax16.plot(time, np.array(d_cp_cg), color='black', label='d_cp_cg', linewidth=2)
        ax16.plot(time, np.array(d_thrust_cg), color='red', label='d_thrust_cg', linewidth=2)
        ax16.set_xlabel('Time [s]', fontsize=16)
        ax16.set_ylabel('Distance [m]', fontsize=16)
        ax16.set_title('Distances over Time', fontsize=18)
        ax16.legend(fontsize=14)
        ax16.grid(True)

        ax17 = plt.subplot(gs[4, 0:2])
        ax17.plot(time, np.array(throttles), color='black', label='Actions', linewidth=2)
        ax17.set_xlabel('Time [s]', fontsize=16)
        ax17.set_ylabel('Throttle [-]', fontsize=16)
        ax17.set_title('Throttle over Time', fontsize=18)
        ax17.legend(fontsize=14)
        ax17.grid(True)

        ax18 = plt.subplot(gs[4, 2:4])
        ax18.plot(time, np.array(gimbal_angle_deg), color='red', label='Gimbal Angle', linewidth=2)
        ax18.set_xlabel('Time [s]', fontsize=16)
        ax18.set_ylabel('Gimbal Angle [deg]', fontsize=16)
        ax18.set_title('Gimbal Angle over Time', fontsize=18)
        ax18.legend(fontsize=14)
        ax18.grid(True)

        plt.savefig(agent.save_path + 'Simulation.png')
        plt.close()

        # Reference tracking plot
        plt.rcParams.update({'font.size': 14})
        
        reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
        xr_array = []
        yr_array = []
        vxr_array = []
        vyr_array = []
        error_x_array = []
        error_y_array = []
        for i, t in enumerate(time):
            xr, yr, vxr, vyr, m = reference_trajectory_func(t)
            xr_array.append(xr)
            yr_array.append(yr)
            vxr_array.append(vxr)
            vyr_array.append(vyr)
            error_x_array.append(abs(x_array[i] - xr))
            error_y_array.append(abs(y_array[i] - yr))

        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(time, np.array(x_array), color='blue', label='agent', linewidth=2)
        ax1.plot(time, np.array(xr_array), color='red', label='reference', linewidth=2)
        ax1.set_xlabel('Time [s]', fontsize=16)
        ax1.set_ylabel('x [m]', fontsize=16)
        ax1.set_title('Position x over Time', fontsize=18)
        ax1.legend(fontsize=14)
        ax1.grid(True)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(time, np.array(y_array), color='green', label='agent', linewidth=2)
        ax2.plot(time, np.array(yr_array), color='purple', label='reference', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=16)
        ax2.set_ylabel('y [m]', fontsize=16)
        ax2.set_title('Position y over Time', fontsize=18)
        ax2.legend(fontsize=14)
        ax2.grid(True)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(time, np.array(vx_array), color='blue', label='agent', linewidth=2)
        ax3.plot(time, np.array(vxr_array), color='red', label='reference', linewidth=2)
        ax3.set_xlabel('Time [s]', fontsize=16)
        ax3.set_ylabel('vx [m/s]', fontsize=16)
        ax3.set_title('Velocity vx over Time', fontsize=18)
        ax3.legend(fontsize=14)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(time, np.array(vy_array), color='blue', label='agent', linewidth=2)
        ax4.plot(time, np.array(vyr_array), color='red', label='reference', linewidth=2)
        ax4.set_xlabel('Time [s]', fontsize=16)
        ax4.set_ylabel('vy [m/s]', fontsize=16)
        ax4.set_title('Velocity vy over Time', fontsize=18)
        ax4.legend(fontsize=14)
        ax4.grid(True)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(time, np.array(error_x_array), color='blue', label='agent', linewidth=2)
        ax5.set_xlabel('Time [s]', fontsize=16)
        ax5.set_ylabel('error_x [m]', fontsize=16)
        ax5.set_title('Error x over Time', fontsize=18)
        ax5.grid(True)

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(time, np.array(error_y_array), color='blue', label='agent', linewidth=2)
        ax6.set_xlabel('Time [s]', fontsize=16)
        ax6.set_ylabel('error_y [m]', fontsize=16)
        ax6.set_title('Error y over Time', fontsize=18)
        ax6.grid(True)
        
        
        plt.savefig(agent.save_path + 'ReferenceTracking.png')
        plt.close()
    else:
        print("Warning: No simulation data collected. The simulation may have terminated immediately.")



# Evolutionary Algorithms
def test_agent_interaction_evolutionary_algorithms(evolutionary_algorithm_env,
                                                   save_path):
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []
    theta_array = []
    theta_dot_array = []
    gamma_array = []
    alpha_array = []
    mass_array = []
    acceleration_x_component_thrust = []
    acceleration_y_component_thrust = []
    acceleration_x_component_drag = []
    acceleration_y_component_drag = []
    acceleration_x_component_gravity = []
    acceleration_y_component_gravity = []
    acceleration_x_component_lift = []
    acceleration_y_component_lift = []
    acceleration_x_component = []
    acceleration_y_component = []

    mach_number = []
    CLs = []
    CDs = []

    moments = []
    moments_thrust = []
    moments_aero = []
    inertia = []

    gimbal_angle_deg = []
    throttles_gimballed = []
    throttles_non_gimballed = []

    d_cp_cg = []
    d_thrust_cg = []

    time = []
    done_or_truncated = False
    state = evolutionary_algorithm_env.env.reset()
    while not done_or_truncated:
        actions = evolutionary_algorithm_env.actor.forward(state)
        state, reward, done, truncated, info = evolutionary_algorithm_env.env.step(actions)
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

        time.append(t)

        acceleration_dict = info['acceleration_dict']
        acceleration_x_component_thrust.append(acceleration_dict['acceleration_x_component_thrust'])
        acceleration_y_component_thrust.append(acceleration_dict['acceleration_y_component_thrust'])
        acceleration_x_component_drag.append(acceleration_dict['acceleration_x_component_drag'])
        acceleration_y_component_drag.append(acceleration_dict['acceleration_y_component_drag'])
        acceleration_x_component_gravity.append(acceleration_dict['acceleration_x_component_gravity'])
        acceleration_y_component_gravity.append(acceleration_dict['acceleration_y_component_gravity'])
        acceleration_x_component_lift.append(acceleration_dict['acceleration_x_component_lift'])
        acceleration_y_component_lift.append(acceleration_dict['acceleration_y_component_lift'])
        acceleration_x_component.append(acceleration_dict['acceleration_x_component'])
        acceleration_y_component.append(acceleration_dict['acceleration_y_component'])
        mach_number.append(info['mach_number'])
        CLs.append(info['CL'])
        CDs.append(info['CD'])
        moments.append(info['moment_dict']['moments_z'])
        moments_thrust.append(info['moment_dict']['thrust_moments_z'])
        moments_aero.append(info['moment_dict']['aero_moements_z'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])


        gimbal_angle_deg.append(info['gimbal_angle_deg'])
        throttles_gimballed.append(info['throttle_gimballed'])
        throttles_non_gimballed.append(info['throttle_non_gimballed'])


    if len(time) > 0:
        plt.rcParams.update({'font.size': 14})
        
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1], hspace=0.6, wspace=0.3)

        # Subplot 1: x vs Time
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(time, x_array, color='blue', linewidth=2)
        ax1.set_xlabel('Time [s]', fontsize=16)
        ax1.set_ylabel('x [m]', fontsize=16)
        ax1.set_title('Position x over Time', fontsize=18)
        ax1.grid(True)

        # Subplot 2: y vs Time
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(time, np.array(y_array), color='green', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=16)
        ax2.set_ylabel('y [m]', fontsize=16)
        ax2.set_title('Position y over Time', fontsize=18)
        ax2.grid(True)

        # Subplot 3: vx vs Time
        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(time, vx_array, color='red', linewidth=2)
        ax3.set_xlabel('Time [s]', fontsize=16)
        ax3.set_ylabel('vx [m/s]', fontsize=16)
        ax3.set_title('Velocity vx over Time', fontsize=18)
        ax3.grid(True)

        # Subplot 4: vy vs Time
        ax4 = plt.subplot(gs[0, 3])
        ax4.plot(time, np.array(vy_array), color='purple', linewidth=2)
        ax4.set_xlabel('Time [s]', fontsize=16)
        ax4.set_ylabel('vy [m/s]', fontsize=16)
        ax4.set_title('Velocity vy over Time', fontsize=18)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 0])
        ax5.plot(time, np.rad2deg(theta_array), label='theta', color='orange', linewidth=2)
        ax5.plot(time, np.rad2deg(gamma_array), label='gamma', color='cyan', linewidth=2)
        ax5.plot(time, np.rad2deg(alpha_array), label='alpha', color='magenta', linewidth=2)
        ax5.set_xlabel('Time [s]', fontsize=16)
        ax5.set_ylabel('Angle [deg]', fontsize=16)
        ax5.set_title('Euler Angles over Time', fontsize=18)
        ax5.legend(fontsize=14)
        ax5.grid(True)


        ax6 = plt.subplot(gs[1, 1])
        ax6.plot(time, np.rad2deg(theta_dot_array), color='brown', linewidth=2)
        ax6.set_xlabel('Time [s]', fontsize=16)
        ax6.set_ylabel('theta_dot [deg/s]', fontsize=16)
        ax6.set_title('Theta Dot over Time', fontsize=18)
        ax6.grid(True)

        # Calculate gamma_dot_array only if we have data
        if len(gamma_array) > 1:  # Need at least 2 points for gradient
            gamma_dot_array = np.rad2deg(np.gradient(gamma_array, time))
            ax7 = plt.subplot(gs[1, 2])
            ax7.plot(time, gamma_dot_array, color='yellow', linewidth=2)
            ax7.set_xlabel('Time [s]', fontsize=16)
            ax7.set_ylabel('gamma_dot [deg/s]', fontsize=16)
            ax7.set_title('Gamma Dot over Time', fontsize=18)
            ax7.grid(True)
        else:
            # Create empty plot if not enough data for gradient
            ax7 = plt.subplot(gs[1, 2])
            ax7.set_xlabel('Time [s]', fontsize=16)
            ax7.set_ylabel('gamma_dot [deg/s]', fontsize=16)
            ax7.set_title('Gamma Dot over Time (insufficient data)', fontsize=18)
            ax7.grid(True)

        ax8 = plt.subplot(gs[1, 3])
        ax8.plot(time, np.array(mass_array)/1000, color='black', label='Mass', linewidth=2)
        ax8.set_xlabel('Time [s]', fontsize=16)
        ax8.set_ylabel('Mass [ton]', fontsize=16)
        ax8.set_title('Mass over Time', fontsize=18)
        ax8.legend(fontsize=14)
        ax8.grid(True)

        ax9 = plt.subplot(gs[2, 0])
        ax9.plot(time, np.array(mach_number), color='black', label='Mach Number', linewidth=2)
        ax9.axhline(y=0.8, color='r', linestyle='--')
        ax9.axhline(y=1.2, color='r', linestyle='--')
        ax9.set_xlabel('Time [s]', fontsize=16)
        ax9.set_ylabel('Mach Number [-]', fontsize=16)
        ax9.set_title('Mach number over Time', fontsize=18)
        ax9.grid(True)

        ax10 = plt.subplot(gs[2, 1])
        ax10.plot(time, np.array(acceleration_x_component), color='black', label='Total', linestyle='--', linewidth=2)
        ax10.plot(time, np.array(acceleration_x_component_thrust), color='red', label='Thrust', linestyle='-.', linewidth=2)
        ax10.plot(time, np.array(acceleration_x_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax10.set_xlabel('Time [s]', fontsize=16)
        ax10.set_ylabel('Acceleration x [m/s^2]', fontsize=16)
        ax10.set_title('Rightward Thrust over Time', fontsize=18)
        ax10.legend(fontsize=14)
        ax10.grid(True)

        ax11 = plt.subplot(gs[2, 2])
        ax11.plot(time, np.array(acceleration_y_component), color='black', label='Total', linewidth=2)
        ax11.plot(time, np.array(acceleration_y_component_thrust), color='red', label='Thrust', linewidth=2)
        ax11.plot(time, np.array(acceleration_y_component_drag), color='blue', label='Drag', linewidth=1.5)
        ax11.plot(time, np.array(acceleration_y_component_gravity), color='green', label='Gravity')
        ax11.plot(time, np.array(acceleration_y_component_lift), color='purple', label='Lift', linewidth=1.5)
        ax11.set_xlabel('Time [s]', fontsize=16)
        ax11.set_ylabel('Acceleration y [m/s^2]', fontsize=16)
        ax11.set_title('Upward Thrust over Time', fontsize=18)
        ax11.legend(fontsize=14)
        ax11.grid(True)

        ax12 = plt.subplot(gs[2, 3])
        ax12.plot(time, np.array(CLs), color='black', label='CL', linewidth=2)
        ax12.set_xlabel('Time [s]', fontsize=16)
        ax12.set_ylabel('CL [-]', fontsize=16)
        ax12.set_title('Lift Coefficient over Time', fontsize=18)
        ax12.grid(True)

        ax13 = plt.subplot(gs[3, 0])
        ax13.plot(time, np.array(CDs), color='black', label='CD', linewidth=2)
        ax13.set_xlabel('Time [s]', fontsize=16)
        ax13.set_ylabel('CD [-]', fontsize=16)
        ax13.set_title('Drag Coefficient over Time', fontsize=18)
        ax13.grid(True)

        ax14 = plt.subplot(gs[3, 1])
        ax14.plot(time, np.array(moments), color='black', label='Total', linewidth=2)
        ax14.plot(time, np.array(moments_thrust), color='red', label='Thrust', linewidth=2)
        ax14.plot(time, np.array(moments_aero), color='blue', label='Aero', linewidth=1.5)
        ax14.set_xlabel('Time [s]', fontsize=16)
        ax14.set_ylabel('Moments [Nm]', fontsize=16)
        ax14.set_title('Moments over Time', fontsize=18)
        ax14.legend(fontsize=14)
        ax14.grid(True)

        ax15 = plt.subplot(gs[3, 2])
        ax15.plot(time, np.array(inertia), color='black', label='Inertia', linewidth=2)
        ax15.set_xlabel('Time [s]', fontsize=16)
        ax15.set_ylabel('Inertia [kg*m^2]', fontsize=16)
        ax15.set_title('Inertia over Time', fontsize=18)
        ax15.grid(True)

        ax16 = plt.subplot(gs[3, 3])
        ax16.plot(time, np.array(d_cp_cg), color='black', label='d_cp_cg', linewidth=2)
        ax16.plot(time, np.array(d_thrust_cg), color='red', label='d_thrust_cg', linewidth=2)
        ax16.set_xlabel('Time [s]', fontsize=16)
        ax16.set_ylabel('Distance [m]', fontsize=16)
        ax16.set_title('Distances over Time', fontsize=18)
        ax16.legend(fontsize=14)
        ax16.grid(True)

        ax17 = plt.subplot(gs[4, 0:2])
        ax17.plot(time, np.array(throttles_gimballed), color='black', label='Gimballed', linewidth=2)
        ax17.plot(time, np.array(throttles_non_gimballed), color='red', label='Non-Gimballed', linewidth=2)
        ax17.set_xlabel('Time [s]', fontsize=16)
        ax17.set_ylabel('Throttle [-]', fontsize=16)
        ax17.set_title('Throttle over Time', fontsize=18)
        ax17.legend(fontsize=14)
        ax17.grid(True)

        ax18 = plt.subplot(gs[4, 2:4])
        ax18.plot(time, np.array(gimbal_angle_deg), color='red', label='Gimbal Angle', linewidth=2)
        ax18.set_xlabel('Time [s]', fontsize=16)
        ax18.set_ylabel('Gimbal Angle [deg]', fontsize=16)
        ax18.set_title('Gimbal Angle over Time', fontsize=18)
        ax18.legend(fontsize=14)
        ax18.grid(True)

        plt.savefig(save_path + 'Simulation.png')
        plt.close()

        # Reference tracking plot
        plt.rcParams.update({'font.size': 14})
        
        reference_trajectory_func, final_reference_time = reference_trajectory_lambda()
        xr_array = []
        yr_array = []
        vxr_array = []
        vyr_array = []
        for t in time:
            xr, yr, vxr, vyr, m = reference_trajectory_func(t)
            xr_array.append(xr)
            yr_array.append(yr)
            vxr_array.append(vxr)
            vyr_array.append(vyr)

        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(time, np.array(x_array), color='blue', label='agent', linewidth=2)
        ax1.plot(time, np.array(xr_array), color='red', label='reference', linewidth=2)
        ax1.set_xlabel('Time [s]', fontsize=16)
        ax1.set_ylabel('x [m]', fontsize=16)
        ax1.set_title('Position x over Time', fontsize=18)
        ax1.legend(fontsize=14)
        ax1.grid(True)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(time, np.array(y_array), color='green', label='agent', linewidth=2)
        ax2.plot(time, np.array(yr_array), color='purple', label='reference', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=16)
        ax2.set_ylabel('y [m]', fontsize=16)
        ax2.set_title('Position y over Time', fontsize=18)
        ax2.legend(fontsize=14)
        ax2.grid(True)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(time, np.array(vx_array), color='blue', label='agent', linewidth=2)
        ax3.plot(time, np.array(vxr_array), color='red', label='reference', linewidth=2)
        ax3.set_xlabel('Time [s]', fontsize=16)
        ax3.set_ylabel('vx [m/s]', fontsize=16)
        ax3.set_title('Velocity vx over Time', fontsize=18)
        ax3.legend(fontsize=14)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(time, np.array(vy_array), color='blue', label='agent', linewidth=2)
        ax4.plot(time, np.array(vyr_array), color='red', label='reference', linewidth=2)
        ax4.set_xlabel('Time [s]', fontsize=16)
        ax4.set_ylabel('vy [m/s]', fontsize=16)
        ax4.set_title('Velocity vy over Time', fontsize=18)
        ax4.legend(fontsize=14)
        ax4.grid(True)

        plt.savefig(save_path + 'ReferenceTracking.png')
        plt.close()
    else:
        print("Warning: No simulation data collected. The simulation may have terminated immediately.")