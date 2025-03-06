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
                             print_bool):
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

    actions_res = []
    force_ratio_x = []
    force_ratio_y = []

    d_cp_cg = []
    d_thrust_cg = []

    time = []
    t = 0
    done_or_truncated = False
    agent_state = env.test_model_setup()
    if print_bool:
        print(f'Initial agent_state: {agent_state}')
    while not done_or_truncated:

        actions_arr = agent.select_actions_no_stochatic(agent_state)
        actions = actions_arr[0]    
        actions_res.append(actions)
        agent_state, reward, done, truncated, info = env.step(actions)
        physics_state = info['physics_state']
        done_or_truncated = done or truncated

        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass = physics_state
        
        x_array.append(x)
        y_array.append(y)
        vx_array.append(vx)
        vy_array.append(vy)
        theta_array.append(theta)
        theta_dot_array.append(theta_dot)
        gamma_array.append(gamma)
        alpha_array.append(alpha)
        mass_array.append(mass)

        t += dt
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
        moments.append(info['moment_dict']['moments_y'])
        moments_thrust.append(info['moment_dict']['thrust_moments_y'])
        moments_aero.append(info['moment_dict']['aero_moements_y'])
        inertia.append(info['inertia'])
        d_cp_cg.append(info['d_cp_cg'])
        d_thrust_cg.append(info['d_thrust_cg'])

        ratio_force_gimballed_x = actions * 0.2
        ratio_force_gimballed_y = 1 - abs(ratio_force_gimballed_x)
        force_ratio_x.append(ratio_force_gimballed_x)
        force_ratio_y.append(ratio_force_gimballed_y)

        
        if print_bool:
            print(f'Altitude: {y} m, Done: {done}, Truncated: {truncated}')
            if truncated:
                print(f'truncation id: {env.truncation_id}')
                print(f'agent state: {agent_state}')

    x_terminal_conditions = env.terminal_conditions_bounds[0]
    y_terminal_conditions = env.terminal_conditions_bounds[1]
    vx_terminal_conditions = env.terminal_conditions_bounds[2]
    vy_terminal_conditions = env.terminal_conditions_bounds[3]
    theta_terminal_conditions = np.rad2deg(env.terminal_conditions_bounds[4])
    theta_dot_terminal_conditions = np.rad2deg(env.terminal_conditions_bounds[5])
    gamma_terminal_conditions = np.rad2deg(env.terminal_conditions_bounds[6])
    alpha_terminal_conditions = np.rad2deg(env.terminal_conditions_bounds[7])
    mass_terminal_conditions = env.terminal_conditions_bounds[8]
        
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)

    # Subplot 1: x vs Time
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, x_array, color='blue')
    ax1.axhline(y=x_terminal_conditions[0], color='r', linestyle='--', label='Min', linewidth=0.5)
    ax1.axhline(y=x_terminal_conditions[1], color='r', linestyle='--', label='Max', linewidth=0.5)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('x [m]')
    ax1.set_title('Position x over Time')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: y vs Time
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, np.array(y_array), color='green')
    ax2.axhline(y=y_terminal_conditions[0], color='green', linestyle='-.', label='Min', linewidth=0.5)
    ax2.axhline(y=y_terminal_conditions[1], color='green', linestyle='--', label='Max', linewidth=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Position y over Time')
    ax2.legend()
    ax2.grid(True)

    # Subplot 3: vx vs Time
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(time, vx_array, color='red')
    ax3.axhline(y=vx_terminal_conditions[0], color='red', linestyle='-.', label='Min', linewidth=0.5)
    ax3.axhline(y=vx_terminal_conditions[1], color='red', linestyle='--', label='Max', linewidth=2.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('vx [m/s]')
    ax3.set_title('Velocity vx over Time')
    ax3.legend()
    ax3.grid(True)

    # Subplot 4: vy vs Time
    ax4 = plt.subplot(gs[0, 3])
    ax4.plot(time, np.array(vy_array), color='purple')
    ax4.axhline(y=vy_terminal_conditions[0], color='purple', linestyle='-.', label='Min', linewidth=0.5) 
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('vy [m/s]')
    ax4.set_title('Velocity vy over Time')
    ax4.legend()
    ax4.grid(True)

    ax5 = plt.subplot(gs[1, 0])
    ax5.plot(time, np.rad2deg(theta_array), label='theta', color='orange')
    ax5.plot(time, np.rad2deg(gamma_array), label='gamma', color='cyan')
    ax5.plot(time, np.rad2deg(alpha_array), label='alpha', color='magenta')
    ax5.axhline(y=theta_terminal_conditions[0], color='orange', linestyle='--', label='Min', linewidth=0.5)
    ax5.axhline(y=theta_terminal_conditions[1], color='orange', linestyle='--', label='Max', linewidth=0.5)
    ax5.axhline(y=gamma_terminal_conditions[0], color='cyan', linestyle='--', label='Min', linewidth=0.5)
    ax5.axhline(y=gamma_terminal_conditions[1], color='cyan', linestyle='--', label='Max', linewidth=0.5)
    ax5.axhline(y=alpha_terminal_conditions[0], color='magenta', linestyle='--', label='Min', linewidth=0.5)
    ax5.axhline(y=alpha_terminal_conditions[1], color='magenta', linestyle='--', label='Max', linewidth=0.5)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Angle [deg]')
    ax5.set_title('Euler Angles over Time')
    ax5.legend()
    ax5.grid(True)


    ax6 = plt.subplot(gs[1, 1])
    ax6.plot(time, np.rad2deg(theta_dot_array), color='brown')
    ax6.axhline(y=theta_dot_terminal_conditions[0], color='brown', linestyle='--', label='Min', linewidth=0.5)
    ax6.axhline(y=theta_dot_terminal_conditions[1], color='brown', linestyle='--', label='Max', linewidth=0.5)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('theta_dot [deg/s]')
    ax6.set_title('Theta Dot over Time')
    ax6.legend()
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
    ax8.axhline(y=mass_terminal_conditions[0]/1000, color='black', linestyle='--', label='Min', linewidth=0.5)
    ax8.axhline(y=mass_terminal_conditions[1]/1000, color='black', linestyle='--', label='Max', linewidth=0.5)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Mass [ton]')
    ax8.set_title('Mass over Time')
    ax8.legend()
    ax8.grid(True)

    ax9 = plt.subplot(gs[2, 0])
    ax9.plot(time, np.array(mach_number), color='black', label='Mach Number')
    ax9.axhline(y=0.8, color='r', linestyle='--')
    ax9.axhline(y=1.2, color='r', linestyle='--')
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Mach Number [-]')
    ax9.set_title('Mach number over Time')
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
    ax12.grid(True)

    ax13 = plt.subplot(gs[3, 0])
    ax13.plot(time, np.array(CDs), color='black', label='CD')
    ax13.set_xlabel('Time [s]')
    ax13.set_ylabel('CD [-]')
    ax13.set_title('Drag Coefficient over Time')
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
    ax14.grid(True)

    ax15 = plt.subplot(gs[3, 3])
    ax15.plot(time, np.array(d_cp_cg), color='black', label='d_cp_cg')
    ax15.plot(time, np.array(d_thrust_cg), color='red', label='d_thrust_cg')
    ax15.set_xlabel('Time [s]')
    ax15.set_ylabel('Distance [m]')
    ax15.set_title('Distances over Time')
    ax15.legend()
    ax15.grid(True)

    ax16 = plt.subplot(gs[4, 0:2])
    ax16.plot(time, np.array(actions_res), color='black', label='Actions')
    ax16.set_xlabel('Time [s]')
    ax16.set_ylabel('Actions [-]')
    ax16.set_title('Actions over Time')
    ax16.legend()
    ax16.grid(True)

    ax17 = plt.subplot(gs[4, 2:4])
    ax17.plot(time, np.array(force_ratio_x), color='black', label='Force Ratio X')
    ax17.plot(time, np.array(force_ratio_y), color='red', label='Force Ratio Y')
    ax17.set_xlabel('Time [s]')
    ax17.set_ylabel('Force Ratio [-]')
    ax17.set_title('Force Ratios over Time')
    ax17.legend()

    plt.savefig(agent.save_path + 'Simulation.png')
    plt.close()