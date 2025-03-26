import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load events from file
event_path = "data/pso_saves/endo_ascent_EA_fitting/particle_subswarm_optimisation/run_20250324_175612/events.out.tfevents.1742835372.jonathanvanzyl-HP-ZBook-Studio-G5.73018.2"
acc = EventAccumulator(event_path)
acc.Reload()

# Extract the last scalar value from each scalar tag
scalar_values = []
for tag in acc.Tags().get('scalars', []):
    scalar_values.append(acc.Scalars(tag)[-1].value)

# Compute histogram statistics from corresponding histogram tags
stats = []  # list of (mean, std)
for tag in acc.Tags().get('histograms', []):
    events = acc.Histograms(tag)
    hv = events[-1].histogram_value
    if hv.num > 0:
        mean = hv.sum / hv.num
        variance = hv.sum_squares / hv.num - mean ** 2
        std = math.sqrt(variance)
    else:
        mean, std = 0, 0
    stats.append((mean, std))

# Compute standard deviation distances (assumes one-to-one correspondence)
n_std_list = []
for value, (mean, std) in zip(scalar_values, stats):
    n_std = (value - mean) / std if std != 0 else 0
    n_std_list.append(n_std)

# Define a Gaussian function to fit the histogram
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Create histogram data (density normalized)
counts, bin_edges = np.histogram(n_std_list, bins=10, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit a Gaussian curve to the histogram data
try:
    # Improved initial guess: amplitude is max of counts, mu is mean of n_std_list, sigma is std of n_std_list
    initial_guess = [max(counts), np.mean(n_std_list), np.std(n_std_list)]
    popt, _ = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
except RuntimeError as e:
    print(f"Error in curve fitting: {e}")
    popt = [0, 0, 1]  # Default to a non-informative Gaussian if fitting fails

x_fit = np.linspace(min(n_std_list), max(n_std_list), 100)
y_fit = gaussian(x_fit, *popt)

# Plot histogram and the fitted Gaussian curve
plt.figure(figsize=(10, 6))
plt.hist(n_std_list, bins=10, density=True, alpha=0.6, label="Data")
plt.plot(x_fit, y_fit, 'r--', label="Gaussian Fit")
plt.xlabel("Standard Deviations Away from Mean")
plt.ylabel("Density")
plt.title("Histogram with Gaussian Fit")
plt.legend()
plt.show()
