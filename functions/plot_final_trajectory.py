import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions.params import R_earth

def final_orbit_plotter(states,
                        times,
                        plot_bool=False,
                        full_orbit=False,
                        save_file_path='/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
    if full_orbit:
        save_tag = 'full_orbit'
    else:
        save_tag = 'to_orbit'
    # Plot the final trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[0, :], states[1, :], states[2, :], label='Trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.set_title('Final Trajectory')
    plt.savefig(f'{save_file_path}/{save_tag}.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    # Plot the altitude, velocity, and mass
    altitude = np.linalg.norm(states[:3, :], axis=0) - R_earth
    velocity = np.linalg.norm(states[3:6, :], axis=0)
    mass = states[6, :]
    # tall and not wide
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(times, altitude/1000)
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [km]')
    plt.grid()
    plt.subplot(1,3,2)
    plt.plot(times, velocity/1000)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [km/s]')
    plt.grid()
    plt.subplot(1,3,3)
    plt.plot(times, mass)
    plt.xlabel('Time [s]')
    plt.ylabel('Mass [kg]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{save_file_path}/{save_tag}_altitude_velocity_mass.png')
    if plot_bool:
        plt.show()
    else:
        plt.close()

    # Write data to file
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'm']
    data_dict = {'time': times}
    for name, state in zip(state_names, states):
        data_dict[name] = state

    df = pd.DataFrame(data_dict)
    filename = f'data/{save_tag}.csv'
    df.to_csv(filename, index=False)

    print(f"Data successfully written to {filename}")
