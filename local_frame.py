import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

def plot_azimuth_altitude(altitudes_m,
                          times,
                          headings_deg,
                          save_path,
                          start_times=None,
                          show_bool = False):
    def generate_ticks(max_altitude_km):
        """Generate appropriate ticks for the given maximum altitude"""
        # Round up max altitude to next multiple of 10
        max_tick = np.ceil(max_altitude_km / 10) * 10
        # Create 5-7 ticks (depending on max altitude)
        num_ticks = min(max(5, int(max_tick/20) + 1), 7)
        return np.linspace(0, max_tick, num_ticks)

    # Convert degrees to radians for the polar plot
    headings_rad = np.radians(headings_deg)

    # Create Polar Plot with larger figure size
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    
    # Plot flight path
    ax.plot(headings_rad, altitudes_m/1000, label="Flight Path")  # Convert to km

    # If start times are provided, add circular lines for each phase
    if start_times is not None:
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        phase_names = ['Vertical Rising', 'Gravity Turn', 'Endo Coasting', 'Exo Propelled', 'Exo Coasting']
        
        # Find indices corresponding to start times
        start_indices = []
        for start_time in start_times:
            idx = np.argmin(np.abs(times - start_time))
            start_indices.append(idx)
        
        # Plot circular lines for each phase start
        for i, idx in enumerate(start_indices):
            if idx < len(headings_rad):  # Check if index is valid
                # Create a circle at the altitude where the phase starts
                phase_altitude = altitudes_m[idx]/1000
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(theta, [phase_altitude]*100,
                       color=colors[i], linestyle='--', 
                       label=phase_names[i])

    # Labels and Grid
    ax.set_theta_zero_location('N')  # 0° at North
    ax.set_theta_direction(-1)  # Clockwise angles

    # Set the positions for the cardinal and intercardinal directions
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # Degrees
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticks(np.radians(angles))
    ax.set_xticklabels(labels)

    # Increase spacing of direction labels
    ax.tick_params(axis='x', pad=20)

    # Set radial ticks and labels
    max_altitude_km = np.max(altitudes_m) / 1000
    r_ticks = generate_ticks(max_altitude_km)
    ax.set_rticks(r_ticks)

    # Hide default radial labels
    ax.set_yticklabels([])

    # Add altitude labels (right side)
    for r in r_ticks:
        if r > 0:  # Skip the origin
            ax.text(np.radians(90), r, f'{int(r)} km', 
                    ha='right', va='center',
                    transform=ax.transData)

    # Add time labels (left side)
    time_at_ticks = np.interp(r_ticks, altitudes_m/1000, times)
    for r, t in zip(r_ticks, time_at_ticks):
        if r > 0:  # Skip the origin
            ax.text(np.radians(270), r, f'{int(t)}s', 
                    ha='left', va='center', 
                    transform=ax.transData)

    # Add labels using figure coordinates
    fig.text(0.55, 0.45, 'Altitude [km]', ha='center', va='bottom', fontweight='bold')
    fig.text(0.25, 0.45, 'Time [s]', ha='center', va='bottom', fontweight='bold')

    # Add legend
    if start_times is not None:
        ax.legend(bbox_to_anchor=(1.2, 0.5), loc='center left')

    # Set title with more padding
    plt.title("Azimuth-Altitude Plot", pad=25, y=1.05)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    if show_bool:
        plt.show()
    else:
        plt.close()

    

# ECI to ECEF
def Rz(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def ECI_to_ECEF(state_vectors, times, earth_rotation_angle_init):

    # State vector is a list of [rx, ry, rz, vx, vy, vz, m]
    w_earth = 2 * np.pi / 86164  # Earth angular velocity [rad/s]
    w_earth_ECI = np.array([0, 0, w_earth])

    # Find dt
    dt_vector = np.diff(times)
    earth_rotation_angle = np.zeros(len(times))
    for i in range(len(times)):
        if i == 0:
            earth_rotation_angle[i] = earth_rotation_angle_init
        else:
            earth_rotation_angle[i] = earth_rotation_angle[i-1] + w_earth * dt_vector[i-1]
    earth_rotation_angle_final = earth_rotation_angle[-1]
    # Initialize transformed state vectors
    state_vectors_ECEF = np.zeros_like(state_vectors)
    state_vectors_ECEF[6, :] = state_vectors[6, :]  # Mass stays the same
    
    # Transform position and velocity for each time step
    for i in range(len(times)):
        # Get rotation matrix for current time
        R = Rz(earth_rotation_angle[i])
        
        # Transform position (first 3 components)
        state_vectors_ECEF[0:3, i] = R @ state_vectors[0:3, i]
        
        # Transform velocity (components 3-6)
        # v_ECEF = R @ v_ECI - omega_earth_ECI × r_ECEF
        r_ECEF = state_vectors_ECEF[0:3, i]
        v_ECI = state_vectors[3:6, i]
        w_earth_ECEF = R @ w_earth_ECI  # Rotate Earth's angular velocity into ECEF frame
        
        w_cross_r = np.cross(w_earth_ECEF, r_ECEF)
        state_vectors_ECEF[3:6, i] = R @ v_ECI - w_cross_r

    # Check state shape
    assert state_vectors_ECEF.shape == state_vectors.shape
    
    return state_vectors_ECEF, earth_rotation_angle_final


# ECEF to XYZ
def ecef_to_enu_matrix(phi0, lam0):
    # X->E, Y->U, Z->N (East, North, Up)
    # phi0 is latitude, lam0 is longitude
    return np.array([
        [-np.sin(lam0),              np.cos(lam0),              0],
        [ np.cos(phi0)*np.cos(lam0), np.cos(phi0)*np.sin(lam0), np.sin(phi0)],
        [-np.sin(phi0)*np.cos(lam0), -np.sin(phi0)*np.sin(lam0), np.cos(phi0)]
    ])

def eci_to_local_xyz(state_vectors_ECI, times, earth_rotation_angle_init):
    # 1) Convert ECI -> ECEF
    state_vectors_ECEF, earth_rotation_angle_final = ECI_to_ECEF(state_vectors_ECI, times, earth_rotation_angle_init)
    
    # 2) ECEF of launch site = position at t=0
    r0_ECEF = state_vectors_ECEF[0:3, 0]  # site's ECEF coords

    # 3) Determine launch site lat/lon
    phi0 = 5.2 * np.pi / 180                            # Kourou latitude [rad] - launch altitude
    lam0 = 0                                            # Kourou longitude [rad] - launch altitude

    # 4) Build rotation matrix ECEF->ENU
    R_ecef_to_enu = ecef_to_enu_matrix(phi0, lam0)

    # 5) Transform
    state_vectors_local = np.zeros_like(state_vectors_ECEF)
    for i in range(len(times)):
        r_ECEF = state_vectors_ECEF[0:3, i]
        delta_r_ECEF = r_ECEF - r0_ECEF
        state_vectors_local[0:3, i] = R_ecef_to_enu @ delta_r_ECEF
        v_ECEF = state_vectors_ECEF[3:6, i]
        v_local = R_ecef_to_enu @ v_ECEF  # local velocity
        state_vectors_local[3:6, i] = v_local
    # Mass stays the same
    state_vectors_local[6, :] = state_vectors_ECEF[6, :]

    # Check state shape
    assert state_vectors_local.shape == state_vectors_ECEF.shape

    return state_vectors_local, earth_rotation_angle_final

def calculate_flight_path_angles(vx_s, vy_s, vz_s):
    flight_path_angle = np.arctan2(vy_s, np.sqrt(vx_s**2 + vz_s**2))
    flight_path_angle_deg = np.rad2deg(flight_path_angle)
    return flight_path_angle_deg

def calculate_heading_angles(vx_s, vy_s, vz_s, times):
    # No heading angle if vy_s = 0 and vx_s = 0
    heading_angle = np.arctan2(vx_s, vz_s)
    heading_angle_deg = np.rad2deg(heading_angle)

    times_with_heading = []
    heading_angle_deg_with_heading = []
    indices_with_heading = []
    for i, vz in enumerate(vz_s):
        vx = vx_s[i]
        if (vz != 0 and vx != 0):
            times_with_heading.append(times[i])
            if heading_angle_deg[i] == -180:
                heading_angle_deg_with_heading.append(180)
            else:
                heading_angle_deg_with_heading.append(heading_angle_deg[i])
            indices_with_heading.append(i)
    times_with_heading = np.array(times_with_heading)
    heading_angle_deg_with_heading = np.array(heading_angle_deg_with_heading)
    
    return heading_angle_deg_with_heading, times_with_heading, indices_with_heading

def pre_process_heading_plot(state_vectors_local,
                             times,
                             save_path_aximuth_altitude,
                             start_times=None,
                             show_bool = False):
    x = state_vectors_local[0, :]  # East
    y = state_vectors_local[1, :]  # Up
    z = state_vectors_local[2, :]  # North
    vx = state_vectors_local[3, :]
    vy = state_vectors_local[4, :]
    vz = state_vectors_local[5, :]
    heading_angle_deg, times_with_heading, indices_with_heading = calculate_heading_angles(vx, vy, vz, times)

    altitudes_with_heading_m = y[indices_with_heading]

    plot_azimuth_altitude(altitudes_with_heading_m,
                          times_with_heading,
                          heading_angle_deg,
                          save_path_aximuth_altitude,
                          start_times,
                          show_bool)

def plot_xyz(state_vectors_local,
             times,
             save_path_plot,
             start_times = None,
             show_bool = False):
    x = state_vectors_local[0, :]
    y = state_vectors_local[1, :]
    z = state_vectors_local[2, :]
    vx = state_vectors_local[3, :]
    vy = state_vectors_local[4, :]
    vz = state_vectors_local[5, :]
    m = state_vectors_local[6, :]
    gamma_deg = calculate_flight_path_angles(vx, vy, vz)    

    heading_angle_deg_with_heading, times_with_heading, indices_with_heading = calculate_heading_angles(vx, vy, vz, times)

    if start_times is None:
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs[0, 0].plot(times, x)
        axs[0, 0].set_title('East from Kourou')
        axs[0, 0].set_ylabel('x [m]')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        axs[1, 0].plot(times, y)
        axs[1, 0].set_title('Up from Kourou i.e. Altitude')
        axs[1, 0].set_ylabel('y [m]')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].grid()

        axs[2, 0].plot(times, z)
        axs[2, 0].set_title('North from Kourou')
        axs[2, 0].set_ylabel('z [m]')
        axs[2, 0].set_xlabel('Time [s]')
        axs[2, 0].grid()

        axs[0, 1].plot(times, vx)   
        axs[0, 1].set_title('East velocity')
        axs[0, 1].set_ylabel('vx [m/s]')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].grid()

        axs[1, 1].plot(times, vy)
        axs[1, 1].set_title('Up (vertical) velocity')
        axs[1, 1].set_ylabel('vy [m/s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].grid()

        axs[2, 1].plot(times, vz)
        axs[2, 1].set_title('North velocity')
        axs[2, 1].set_ylabel('vz [m/s]')
        axs[2, 1].set_xlabel('Time [s]')
        axs[2, 1].grid()

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

        axs[2, 2].plot(times_with_heading, heading_angle_deg_with_heading)
        axs[2, 2].set_title('Heading Angle')
        axs[2, 2].set_ylabel('Heading Angle [deg]')
        axs[2, 2].set_xlabel('Time [s]')
        axs[2, 2].grid()

        plt.tight_layout()
        plt.savefig(save_path_plot)
        if show_bool:
            plt.show()
        else:
            plt.close()
    else:
        # Then plot vertical lines for each flight phase start time
        fig, axs = plt.subplots(4, 3, figsize=(10, 10))
        
        # Create a line object for the legend
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        # Create plots with vertical lines
        axs[0, 0].plot(times, x)
        for i, start_time in enumerate(start_times):
            axs[0, 0].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[0, 0].set_title('East from Kourou')
        axs[0, 0].set_ylabel('x [m]')
        axs[0, 0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        axs[1, 0].plot(times, y)
        for i, start_time in enumerate(start_times):
            axs[1, 0].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[1, 0].set_title('Up from Kourou i.e. Altitude')
        axs[1, 0].set_ylabel('y [m]')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].grid()

        axs[2, 0].plot(times, z)
        for i, start_time in enumerate(start_times):
            axs[2, 0].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[2, 0].set_title('North from Kourou')
        axs[2, 0].set_ylabel('z [m]')
        axs[2, 0].set_xlabel('Time [s]')
        axs[2, 0].grid()

        axs[0, 1].plot(times, vx)   
        for i, start_time in enumerate(start_times):
            axs[0, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[0, 1].set_title('East velocity')
        axs[0, 1].set_ylabel('vx [m/s]')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].grid()

        axs[1, 1].plot(times, vy)
        for i, start_time in enumerate(start_times):
            axs[1, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[1, 1].set_title('Up (vertical) velocity')
        axs[1, 1].set_ylabel('vy [m/s]')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].grid()
        axs[2, 1].plot(times, vz)
        for i, start_time in enumerate(start_times):
            axs[2, 1].axvline(x=start_time, color=colors[i], linestyle='--')
        axs[2, 1].set_title('North velocity')
        axs[2, 1].set_ylabel('vz [m/s]')
        axs[2, 1].set_xlabel('Time [s]')
        axs[2, 1].grid()

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
    state_vectors_local, earth_rotation_angle = eci_to_local_xyz(states_ECI, 
                                                                times, 
                                                                initial_earth_rotation_angle)
    save_path_plot = f'results/{flight_phase_name}_local.png'
    plot_xyz(state_vectors_local, times, save_path_plot, start_times = None)
    save_path_azimuth = f'results/{flight_phase_name}_azimuth_altitude.png'
    pre_process_heading_plot(state_vectors_local, times, save_path_azimuth, start_times = None)
    return earth_rotation_angle