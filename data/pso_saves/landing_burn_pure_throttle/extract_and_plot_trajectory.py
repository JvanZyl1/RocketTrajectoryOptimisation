import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# from trajectory_data/trajectory.csv extract: time[s],x[m],y[m],vx[m/s],vy[m/s],theta[rad],theta_dot[rad/s],gamma[rad],alpha[rad],mass[kg],mass_propellant[kg]
# from trajectory_data/info_data.csv extract action_info_throttle as throttle
# and dynamic_pressure

# Then plot
# (x, y) | (t, vy) & (t, vx)
# (t, dynamic_pressure) | (t, throttle)

def extract_and_plot_trajectory(run_name=None, output_filename="trajectory_analysis.png"):
    """
    Extract trajectory data from CSV files and create plots.
    
    Parameters:
    run_name (str): Name of the run directory (e.g., 'run_2025-06-08_21-47-31').
                   If None, the latest run will be used.
    output_filename (str): Name of the output plot file to save in the plots directory.
    
    Returns:
    str: Path to the saved plot file
    """
    # Base directory
    run_dir = "data/pso_saves/landing_burn_pure_throttle"
    
    # Get the run directory
    if run_name is None:
        # Get the most recent run directory
        run_folders = [f for f in os.listdir(run_dir) if f.startswith("run_")]
        run_folders.sort(reverse=True)
        run_name = run_folders[0]
    
    run_path = os.path.join(run_dir, run_name)
    
    # Path to trajectory data
    traj_data_path = os.path.join(run_path, "trajectory_data")
    trajectory_csv = os.path.join(traj_data_path, "trajectory.csv")
    info_data_csv = os.path.join(traj_data_path, "info_data.csv")
    
    # Create output directory for plots if it doesn't exist
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load trajectory data
    trajectory_df = pd.read_csv(trajectory_csv)
    info_df = pd.read_csv(info_data_csv)
    
    # Extract required data
    time = trajectory_df['time[s]']
    x = trajectory_df['x[m]']
    y = trajectory_df['y[m]']
    vx = trajectory_df['vx[m/s]']
    vy = trajectory_df['vy[m/s]']
    
    # Extract throttle and dynamic pressure from info_data.csv
    throttle = info_df['action_info_throttle']
    dynamic_pressure = info_df['dynamic_pressure']
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Plot 1: (x, y)
    plt.subplot(2, 2, 1)
    plt.plot(x, y)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')
    plt.grid(True)
    
    # Plot 2: (t, vy) & (t, vx)
    plt.subplot(2, 2, 2)
    plt.plot(time, vx, label='vx')
    plt.plot(time, vy, label='vy')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: (t, dynamic_pressure)
    plt.subplot(2, 2, 3)
    plt.plot(time[:len(dynamic_pressure)], dynamic_pressure)
    plt.xlabel('Time [s]')
    plt.ylabel('Dynamic Pressure')
    plt.title('Dynamic Pressure vs Time')
    plt.grid(True)
    
    # Plot 4: (t, throttle)
    plt.subplot(2, 2, 4)
    plt.plot(time[:len(throttle)], throttle)
    plt.xlabel('Time [s]')
    plt.ylabel('Throttle')
    plt.title('Throttle vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_path}")
    return output_path

# Example usage:
if __name__ == "__main__":
    # Extract and plot the latest run
    extract_and_plot_trajectory()
    
    # Or specify a particular run and filename
    # extract_and_plot_trajectory("run_2025-06-08_21-47-31", "specific_run_analysis.png")