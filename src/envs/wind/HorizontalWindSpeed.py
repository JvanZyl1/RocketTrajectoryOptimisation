import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_horizontal_wind_data():
    file_path = 'data/Wind/horizontal_wind.csv'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Parse the header to get percentiles
    header_line = lines[0].strip().split(',')
    percentiles = []
    for item in header_line:
        if item and not item.isspace():
            percentiles.append(item)
    data_lines = lines[2:]
    wind_data = {}
    for percentile in percentiles:
        wind_data[percentile] = {'wind_speed': [], 'altitude_km': []}
    for line in data_lines:
        if not line.strip():  # Skip empty lines
            continue
            
        values = line.strip().split(',')
        if len(values) < len(percentiles) * 2:
            continue  # Skip incomplete lines
            
        for i, percentile in enumerate(percentiles):
            try:
                x_index = i * 2
                y_index = i * 2 + 1
                
                if x_index < len(values) and y_index < len(values):
                    wind_speed = float(values[x_index])
                    altitude_km = float(values[y_index])
                    
                    wind_data[percentile]['wind_speed'].append(wind_speed)
                    wind_data[percentile]['altitude_km'].append(altitude_km)
            except (ValueError, IndexError) as e:
                pass  # Skip invalid values
    # Convert to numpy arrays for easier manipulation
    for percentile in percentiles:
        wind_data[percentile]['wind_speed'] = np.array(wind_data[percentile]['wind_speed'])
        wind_data[percentile]['altitude_km'] = np.array(wind_data[percentile]['altitude_km'])
    return wind_data, percentiles

def compile_horizontal_fixed_wind(percentile):
    # From percentile, an interpolator is created for horizontal wind speed as a function of altitude
    wind_data, _ = extract_horizontal_wind_data()
    
    # Check if the percentile exists directly in the data
    if percentile in wind_data:
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
    else:
        # If not, we need to interpolate between existing percentiles
        wind_speed, altitude_km = interpolate_percentile(wind_data, percentile)
    
    sort_idx = np.argsort(altitude_km)
    altitude_km = altitude_km[sort_idx]
    wind_speed = wind_speed[sort_idx]
    
    # Create the interpolation function
    def wind_at_altitude(altitude_m):
        altitude_km_input = altitude_m / 1000.0
        interpolator = interp1d(
            altitude_km, 
            wind_speed, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(wind_speed[0], wind_speed[-1])  # Use edge values for out-of-bounds
        )
        
        return interpolator(altitude_km_input)
    
    return wind_at_altitude

def interpolate_percentile(wind_data, requested_percentile):
    """
    Interpolate between available percentiles to get wind data for any percentile.
    
    Args:
        wind_data: Dictionary with percentile data
        requested_percentile: The percentile to interpolate (e.g., "53_percentile")
    
    Returns:
        Tuple of (interpolated_wind_speed, altitude_km)
    """
    # Extract the numeric percentile value
    if isinstance(requested_percentile, str) and "_percentile" in requested_percentile:
        try:
            req_perc_value = float(requested_percentile.split('_')[0])
        except ValueError:
            raise ValueError(f"Invalid percentile format: {requested_percentile}")
    else:
        # Assume it's already a numeric value
        try:
            req_perc_value = float(requested_percentile)
            requested_percentile = f"{req_perc_value}_percentile"
        except ValueError:
            raise ValueError(f"Invalid percentile value: {requested_percentile}")
    
    # Get available percentiles and their numeric values
    available_percentiles = list(wind_data.keys())
    available_perc_values = [float(p.split('_')[0]) for p in available_percentiles]
    
    # Check if requested percentile is within range
    if req_perc_value < min(available_perc_values) or req_perc_value > max(available_perc_values):
        raise ValueError(f"Requested percentile {req_perc_value} is outside the available range " 
                         f"({min(available_perc_values)}-{max(available_perc_values)})")
    
    # Find the two surrounding percentiles
    idx = np.searchsorted(available_perc_values, req_perc_value)
    if idx == 0:
        # This shouldn't happen due to the range check above
        lower_perc = available_percentiles[0]
        upper_perc = available_percentiles[0]
        weight = 1.0
    elif idx == len(available_perc_values):
        # This shouldn't happen due to the range check above
        lower_perc = available_percentiles[-1]
        upper_perc = available_percentiles[-1]
        weight = 0.0
    else:
        lower_perc = available_percentiles[idx-1]
        upper_perc = available_percentiles[idx]
        
        # Calculate interpolation weight
        lower_val = available_perc_values[idx-1]
        upper_val = available_perc_values[idx]
        weight = (req_perc_value - lower_val) / (upper_val - lower_val)
    
    # Get altitude values from both percentiles and find common altitudes
    lower_alt = wind_data[lower_perc]['altitude_km']
    upper_alt = wind_data[upper_perc]['altitude_km']
    
    # Combine altitudes and remove duplicates
    all_altitudes = np.unique(np.concatenate([lower_alt, upper_alt]))
    
    # Create interpolators for both percentiles
    lower_interp = interp1d(lower_alt, wind_data[lower_perc]['wind_speed'], 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    upper_interp = interp1d(upper_alt, wind_data[upper_perc]['wind_speed'],
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate wind speeds at all altitude points
    lower_speeds = lower_interp(all_altitudes)
    upper_speeds = upper_interp(all_altitudes)
    
    # Linearly interpolate between percentiles
    interpolated_speeds = lower_speeds * (1 - weight) + upper_speeds * weight
    
    return interpolated_speeds, all_altitudes

def plot_horizontal_fixed_wind():
    wind_data, percentiles = extract_horizontal_wind_data()

    # Create interpolation functions for each percentile
    wind_interpolators = {}
    for percentile in percentiles:
        wind_interpolators[percentile] = compile_horizontal_fixed_wind(percentile)

    # Create a plot with all percentiles together
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, percentile in enumerate(percentiles):
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
        altitude_m = altitude_km * 1000
        min_alt_m = np.min(altitude_m)
        max_alt_m = np.max(altitude_m)
        altitude_grid_m = np.linspace(min_alt_m, max_alt_m, 1000)
        interpolator = wind_interpolators[percentile]
        interpolated_wind_speed = interpolator(altitude_grid_m)
        
        # Plot the original data points and the interpolated curve
        label = f"{percentile}"
        color = colors[i % len(colors)]
        plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.5, color=color)
        plt.plot(interpolated_wind_speed, altitude_grid_m, '-', linewidth=2, label=label, color=color)
    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.title('Horizontal Wind Speed vs Altitude - All Percentiles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_all_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate plots for each percentile
    plt.figure(figsize=(15, 10))
    for i, percentile in enumerate(percentiles):
        plt.subplot(2, 3, i+1)
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
        altitude_m = altitude_km * 1000
        min_alt_m = np.min(altitude_m)
        max_alt_m = np.max(altitude_m)
        altitude_grid_m = np.linspace(min_alt_m, max_alt_m, 1000)
        
        interpolator = wind_interpolators[percentile]
        interpolated_wind_speed = interpolator(altitude_grid_m)
        
        # Plot the original data points and the interpolated curve
        color = colors[i % len(colors)]
        plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.7, label='Data points', color=color)
        plt.plot(interpolated_wind_speed, altitude_grid_m, '-', linewidth=2, label='Interpolated', color=color)
        
        plt.xlabel('Wind Speed (m/s)', fontsize=10)
        plt.ylabel('Altitude (m)', fontsize=10)
        plt.title(f"{percentile}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_individual_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    # Example of using the interpolation function
    print("Example wind speeds at different altitudes (50_percentile):")
    test_altitudes = [1000, 5000, 10000, 20000, 50000]  # meters
    for alt in test_altitudes:
        wind_speed = wind_interpolators['50_percentile'](alt)
        print(f"Altitude: {alt} m, Wind Speed: {wind_speed:.2f} m/s")

# Create a test function to demonstrate the percentile interpolation
def test_percentile_interpolation():
    # Test with both included and interpolated percentiles
    test_percentiles = ['50_percentile', '53_percentile', '75_percentile', '85_percentile', '90_percentile']
    
    # Create interpolators for each percentile
    interpolators = {}
    for percentile in test_percentiles:
        try:
            interpolators[percentile] = compile_horizontal_fixed_wind(percentile)
            print(f"Successfully created interpolator for {percentile}")
        except Exception as e:
            print(f"Error creating interpolator for {percentile}: {e}")
    
    # Test the interpolators at a few altitudes
    test_altitudes = [1000, 10000, 50000]  # meters
    
    for percentile, interpolator in interpolators.items():
        print(f"\nWind speeds for {percentile}:")
        for altitude in test_altitudes:
            try:
                wind_speed = interpolator(altitude)
                print(f"  At {altitude} m: {wind_speed:.2f} m/s")
            except Exception as e:
                print(f"  Error at {altitude} m: {e}")

def plot_horizontal_wind():
    # Load wind data
    wind_data, percentiles = extract_horizontal_wind_data()
    original_percentiles = percentiles.copy()
    
    # Add some interpolated percentiles
    all_percentiles = original_percentiles.copy()
    all_percentiles.extend(['53_percentile', '60_percentile', '80_percentile', '85_percentile'])
    
    # Create a plot with all percentiles together
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black', 'cyan', 'magenta']
    for i, percentile in enumerate(all_percentiles):
        try:
            # Get the interpolator for this percentile
            interpolator = compile_horizontal_fixed_wind(percentile)
            
            # For original percentiles, plot the actual data points
            if percentile in original_percentiles:
                wind_speed = wind_data[percentile]['wind_speed']
                altitude_km = wind_data[percentile]['altitude_km']
                altitude_m = altitude_km * 1000
                
                # Plot the original data points
                color = colors[i % len(colors)]
                plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.5, color=color)
            
            # For all percentiles, plot the interpolated curve
            # Create a dense altitude grid for smooth plotting (in meters)
            max_alt_m = 80000  # Set a reasonable maximum altitude
            altitude_grid_m = np.linspace(0, max_alt_m, 1000)
            
            # Get interpolated wind speeds using our function
            interpolated_wind_speed = interpolator(altitude_grid_m)
            
            # Plot the interpolated curve
            linestyle = '-' if percentile in original_percentiles else '--'
            label = f"{percentile}"
            color = colors[i % len(colors)]
            plt.plot(interpolated_wind_speed, altitude_grid_m, linestyle, linewidth=2, 
                    label=label, color=color)
        except Exception as e:
            print(f"Error plotting {percentile}: {e}")
    
    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.title('Horizontal Wind Speed vs Altitude - With Interpolated Percentiles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the combined plot
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_all_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()

# Call the functions to test and plot
if __name__ == "__main__":
    test_percentile_interpolation()
    plot_horizontal_wind()
    plot_horizontal_fixed_wind()