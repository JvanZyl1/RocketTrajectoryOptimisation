import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math

def load_velocity_data(file_path='data/TiltAngle/RETALT/RETALT_Vx_Vy.csv'):
    """Load velocity data from CSV file and return cleaned arrays."""
    # Load data with proper handling of space after comma in header
    df = pd.read_csv(file_path)
    
    # Clean column names by stripping whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Extract Vx and Vy columns
    vx = df['X'].values
    vy = df['Y'].values
    
    return vx, vy

def create_flight_path_angle_function():
    """
    Create a function that returns flight path angle (in radians) as a function of Vy.
    
    Returns:
        function: A function that takes Vy as input and returns the flight path angle in radians.
    """
    # Load velocity data
    vx_raw, vy_raw = load_velocity_data()
    
    # Clip to minimum 0 for both Vx and Vy
    vx = np.maximum(vx_raw, 0)
    vy = np.maximum(vy_raw, 0)
    
    # Calculate flight path angles for each point
    # Flight path angle is arctan(Vy/Vx)
    flight_path_angles = np.arctan2(vy, vx)
    
    # Create interpolation function for flight path angle as a function of Vy
    # We need to ensure Vy is monotonically increasing for interpolation
    sorted_indices = np.argsort(vy)
    vy_sorted = vy[sorted_indices]
    angles_sorted = flight_path_angles[sorted_indices]
    
    # Remove duplicates if any
    unique_indices = np.concatenate(([True], np.diff(vy_sorted) > 0))
    vy_unique = vy_sorted[unique_indices]
    angles_unique = angles_sorted[unique_indices]
    
    # Create interpolation function
    # Use 'nearest' for extrapolation to avoid unreasonable values
    flight_path_angle_func = interp1d(
        vy_unique, 
        angles_unique, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(angles_unique[0], angles_unique[-1])
    )
    
    return flight_path_angle_func

def get_flight_path_angle(vy):
    """
    Get flight path angle in radians for a given Vy value.
    
    Args:
        vy (float): Vertical velocity component.
        
    Returns:
        float: Flight path angle in radians.
    """
    # Get the function that maps Vy to flight path angle
    angle_func = create_flight_path_angle_function()
    
    # Ensure Vy is non-negative
    vy = max(0, vy)
    
    # Return the flight path angle
    return angle_func(vy)

def get_flight_path_angle_degrees(vy):
    """
    Get flight path angle in degrees for a given Vy value.
    
    Args:
        vy (float): Vertical velocity component.
        
    Returns:
        float: Flight path angle in degrees.
    """
    # Handle the special case where Vy = 0
    if vy <= 100:
        return 90.0
    else:
        # Use the radians function to get the angle, then convert to degrees
        return math.degrees(get_flight_path_angle(vy))

# Example usage
if __name__ == "__main__":
    # Test the function with sample Vy values
    test_vy_values = [0, 1,2,3,100, 110, 120, 130, 140, 150, 200, 300, 400, 500, 600, 700, 800, 900]
    
    print("Vy (m/s) | Flight Path Angle (degrees)")
    print("-" * 40)
    
    for vy in test_vy_values:
        angle_deg = get_flight_path_angle_degrees(vy)
        print(f"{vy:7.1f} | {angle_deg:7.2f}")
