import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the standard polynomial function for reference
def poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

# Define a constrained polynomial function that passes through (0,0)
# This means d=0, so we remove that parameter
def poly3_constrained(x, a, b, c):
    return a*x**3 + b*x**2 + c*x  # No constant term, so it passes through (0,0)

# Load the data
# The CSV has an unusual format - the first row seems to be a header, 
# the second row has column names, and the data starts from row 3
data = pd.read_csv("data/TiltAngle/tilt_reference_ascent_extract.csv", header=None, skiprows=2)

# Rename columns for clarity
data.columns = ['Time', 'TiltAngle']

# Extract time and tilt angle data
time_data = data['Time'].values
tilt_data = data['TiltAngle'].values

# Normalize time data to [0, 1]
time_norm = (time_data - time_data.min()) / (time_data.max() - time_data.min())

# Fit constrained polynomial to the data
popt_constrained, _ = curve_fit(poly3_constrained, time_norm, tilt_data)

# Generate smooth curve for plotting
time_fit_norm = np.linspace(0, 1, 500)
tilt_fit_constrained = poly3_constrained(time_fit_norm, *popt_constrained)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(time_norm, tilt_data, label="Data Points", color="red")
plt.plot(time_fit_norm, tilt_fit_constrained, label="Polynomial Fit", 
         linestyle="--", linewidth=2, color="blue")

# Also plot the origin explicitly to verify constraint
plt.plot(0, 0, 'ro', markersize=8)

plt.xlabel("Normalized time", fontsize=14)
plt.ylabel(r"Tilt Angle ($^\circ$)", fontsize=14)
plt.title("Constrained Tilt Angle vs. Normalized Time", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("data/TiltAngle/tilt_reference_ascent_constrained_fit.png", dpi=300)
plt.close()


def tilt_angle_function(normalized_time):
    a, b, c = popt_constrained
    return a*normalized_time**3 + b*normalized_time**2 + c*normalized_time

def compile_pitch_angle_reference(time_length):
    def pitch_angle_reference(time):
        tilt_angle_deg = tilt_angle_function(time/time_length)
        pitch_angle_radians = math.pi/2 - math.radians(tilt_angle_deg)
        return pitch_angle_radians
    return pitch_angle_reference


    