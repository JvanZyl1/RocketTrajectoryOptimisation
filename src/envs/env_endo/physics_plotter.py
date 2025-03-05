import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    dt = 0.01

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
        moments.append(info['moment_dict']['moments_y'])
        moments_thrust.append(info['moment_dict']['thrust_moments_y'])
        moments_aero.append(info['moment_dict']['aero_moements_y'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])

        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass = state
        t += dt
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

    save_path_physics = os.path.join(os.path.abspath(".."), "results", "figures", "PhysicsGravityTurn", "Simulation_VRandGT.png")
    plt.savefig(save_path_physics)
    plt.close()