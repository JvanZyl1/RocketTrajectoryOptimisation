import math
import numpy as np
import matplotlib.pyplot as plt

                                                                
def transform_eci_state_to_local_frame(state_vectors_ECI, times, initial_earth_rotation_angle):
    # State vectors are [rx, ry, rz, vx, vy, vz, m]
    # State vectors has shape (7, len(times))
    # times has shape (len(times),)
    
    w_earth_east = 2 * math.pi / 86164  # Earth angular velocity [rad/s]
    w_earth = np.array([0, 0, w_earth_east])  # Earth angular velocity [rad/s]
    R_earth = 6378137 # [m]
    positions_ECI = state_vectors_ECI[0:3, :]
    velocities_ECI = state_vectors_ECI[3:6, :]
    masses = state_vectors_ECI[6, :]

    dt = np.diff(times)
    dt = np.insert(dt, 0, 0)
    pos_xyz = np.zeros((3, len(times)))
    vel_xyz = np.zeros((3, len(times)))
    earth_rotation_angle = initial_earth_rotation_angle
    for i in range(len(times)):
        # Transform ECI to ECEF
        earth_rotation_angle = earth_rotation_angle + w_earth_east * dt[i]
        pos_eci = positions_ECI[:, i]
        x_eci = pos_eci[0]
        y_eci = pos_eci[1]
        z_eci = pos_eci[2]
        x_ecef = np.cos(earth_rotation_angle) * x_eci + np.sin(earth_rotation_angle) * y_eci
        y_ecef = -(-np.sin(earth_rotation_angle) * x_eci + np.cos(earth_rotation_angle) * y_eci)
        z_ecef = z_eci

        # Transform ECEF to local frame
        x_local_up = x_ecef - R_earth
        y_local_east = y_ecef
        z_local_north = z_ecef
        pos_xyz[0, i] = x_local_up
        pos_xyz[1, i] = y_local_east
        pos_xyz[2, i] = z_local_north

        # Vel ECI
        if i != 0:
            vel_xyz_x = (pos_xyz[0, i] - pos_xyz[0, i-1]) / dt[i]
            vel_xyz_y = (pos_xyz[1, i] - pos_xyz[1, i-1]) / dt[i]
            vel_xyz_z = (pos_xyz[2, i] - pos_xyz[2, i-1]) / dt[i]
        else:
            vel_xyz_x = 0
            vel_xyz_y = 0
            vel_xyz_z = 0
        vel_xyz[0, i] = vel_xyz_x
        vel_xyz[1, i] = vel_xyz_y
        vel_xyz[2, i] = vel_xyz_z


    states = np.concatenate((pos_xyz,
                             vel_xyz,
                             masses.reshape(1, -1)), axis=0)
    
    
    return states, earth_rotation_angle


def calculate_flight_path_angles(vx_s, vy_s):
    flight_path_angle = np.arctan2(vx_s, vy_s)
    flight_path_angle_deg = np.rad2deg(flight_path_angle)
    return flight_path_angle_deg


def plot_xyz(state_vectors_local,
             times,
             save_path_plot,
             start_times = None,
             show_bool = False):
    x = state_vectors_local[0, :]                   # Up
    y = state_vectors_local[1, :]                   # East
    vx = state_vectors_local[3, :]                  # Up
    vy = state_vectors_local[4, :]                  # East
    m = state_vectors_local[6, :]
    gamma_deg = calculate_flight_path_angles(vx, vy)    

    if start_times is None:
        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        axs[0, 0].plot(times, x)
        axs[0, 0].set_title('Up from Kourou i.e. Altitude')
        axs[0, 0].set_ylabel('x [m]')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        axs[1, 0].plot(times, y)
        axs[1, 0].set_title('East from Kourou')
        axs[1, 0].set_ylabel('y [m]')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].grid()

        axs[0, 1].plot(times, vx)   
        axs[0, 1].set_title('Up (vertical) velocity')
        axs[0, 1].set_ylabel('vx [m/s]')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].grid()

        axs[1, 1].plot(times, vy)
        axs[1, 1].set_title('East velocity')
        axs[1, 1].set_ylabel('vy [m/s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].grid()


        axs[0, 2].plot(times, m/1000)
        axs[0, 2].set_title('Mass')
        axs[0, 2].set_ylabel('m [tonnes]')
        axs[0, 2].set_xlabel('Time [s]')
        axs[0, 2].grid()

        axs[1, 2].plot(times[1:], gamma_deg[1:])
        axs[1, 2].set_title('Flight Path Angle')
        axs[1, 2].set_ylabel('Flight Path Angle [deg]')
        axs[1, 2].set_xlabel('Time [s]')
        axs[1, 2].grid()

        plt.tight_layout()
        plt.savefig(save_path_plot)
        if show_bool:
            plt.show()
        else:
            plt.close()
    else:
        # Then plot vertical lines for each flight phase start time
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        
        # Create a line object for the legend
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        # Create plots with vertical lines
        axs[0, 0].plot(times, x)
        for i, start_time in enumerate(start_times):
            axs[0, 0].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[0, 0].set_title('Up from Kourou i.e. Altitude')
        axs[0, 0].set_ylabel('x [m]')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        axs[1, 0].plot(times, y)
        for i, start_time in enumerate(start_times):
            axs[1, 0].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[1, 0].set_title('East from Kourou')
        axs[1, 0].set_ylabel('y [m]')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].grid()

        axs[0, 1].plot(times, vx)   
        for i, start_time in enumerate(start_times):
            axs[0, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[0, 1].set_title('Up (vertical) velocity')
        axs[0, 1].set_ylabel('vx [m/s]')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].grid()

        axs[1, 1].plot(times, vy)
        for i, start_time in enumerate(start_times):
            axs[1, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[1, 1].set_title('East velocity')
        axs[1, 1].set_ylabel('vy [m/s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].grid()

        axs[3, 1].plot(times, m/1000)
        for i, start_time in enumerate(start_times):
            axs[3, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[3, 1].set_title('Mass')
        axs[3, 1].set_ylabel('m [tonnes]')
        axs[3, 1].set_xlabel('Time [s]')
        axs[3, 1].grid()

        axs[0, 2].plot(times[1:], gamma_deg[1:])
        for i, start_time in enumerate(start_times):
            axs[0, 2].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[0, 2].set_title('Flight Path Angle')
        axs[0, 2].set_ylabel('Flight Path Angle [deg]')
        axs[0, 2].set_xlabel('Time [s]')
        axs[0, 2].grid()
        # axs [3,0] is legend of flight phases
        phase_names = ['Vertical Rising', 'Gravity Turn', 'Endo Coasting', 'Exo Propelled', 'Exo Coasting']
        # Create colored markers for each phase
        markers = [plt.Line2D([0], [0], color=colors[i], linestyle='--', label=name) for i, name in enumerate(phase_names)]
        
        axs[3, 0].legend(handles=markers, 
                        title='Flight Phases',
                        loc='center')
        axs[3, 0].axis('off')

        plt.tight_layout()
        plt.savefig(save_path_plot)
        if show_bool:
            plt.show()
        else:
            plt.close()


def plot_eci_to_local_xyz(states_ECI,
                          times,
                          initial_earth_rotation_angle,
                          flight_phase_name):
    state_vectors_local, earth_rotation_angle = transform_eci_state_to_local_frame(states_ECI, 
                                                                times, 
                                                                initial_earth_rotation_angle)
    save_path_plot = f'results/{flight_phase_name}_local.png'
    plot_xyz(state_vectors_local, times, save_path_plot, start_times = None)
    return earth_rotation_angle