import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import pandas as pd

import sys
sys.path.append('.')


from src.envs.disturbance_generator import VonKarmanFilter

def analyze_parameter_sensitivity():
    # Base parameters
    dt = 0.02
    V = 100.0
    n_samples = 10000
    
    # Parameter ranges to test
    L_values = np.linspace(50, 500, 50)
    sigma_values = np.linspace(0.5, 2.0, 50)
    
    # Initialize results storage
    results = {
        'L': [],
        'sigma': [],
        'mean_velocity': [],
        'std_velocity': [],
        'kurtosis': [],
        'autocorr_lag1': [],
        'psd_slope': []
    }
    
    # Analyze sensitivity to L
    for L in L_values:
        filter = VonKarmanFilter(L=L, sigma=1.0, V=V, dt=dt)
        velocities = np.array([filter.step() for _ in range(n_samples)])
        
        # Calculate metrics
        results['L'].append(L)
        results['mean_velocity'].append(np.mean(velocities))
        results['std_velocity'].append(np.std(velocities))
        results['kurtosis'].append(stats.kurtosis(velocities))
        results['autocorr_lag1'].append(np.corrcoef(velocities[:-1], velocities[1:])[0,1])
        
        # Calculate PSD slope
        f_psd, Pxx = plt.psd(velocities, Fs=1/dt, NFFT=1024, detrend='linear')
        slope = np.polyfit(np.log(f_psd[1:]), np.log(Pxx[1:]), 1)[0]
        results['psd_slope'].append(slope)
    
    # Analyze sensitivity to sigma
    for sigma in sigma_values:
        filter = VonKarmanFilter(L=200.0, sigma=sigma, V=V, dt=dt)
        velocities = np.array([filter.step() for _ in range(n_samples)])
        
        # Calculate metrics
        results['sigma'].append(sigma)
        results['mean_velocity'].append(np.mean(velocities))
        results['std_velocity'].append(np.std(velocities))
        results['kurtosis'].append(stats.kurtosis(velocities))
        results['autocorr_lag1'].append(np.corrcoef(velocities[:-1], velocities[1:])[0,1])
        
        # Calculate PSD slope
        f_psd, Pxx = plt.psd(velocities, Fs=1/dt, NFFT=1024, detrend='linear')
        slope = np.polyfit(np.log(f_psd[1:]), np.log(Pxx[1:]), 1)[0]
        results['psd_slope'].append(slope)
    
    # Plot results
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(results['L'], results['std_velocity'][:len(L_values)], 'o-', color='blue', linewidth=4)
    ax1.set_xlabel('Length Scale (m)', fontsize=20)
    ax1.set_ylabel('Standard Deviation (m/s)', fontsize=20)
    ax1.set_title('Sensitivity to Length Scale', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(results['L'], results['autocorr_lag1'][:len(L_values)], 'o-', color='blue', linewidth=4)
    ax2.set_xlabel('Length Scale (m)', fontsize=20)
    ax2.set_ylabel('Autocorrelation (lag 1)', fontsize=20)
    ax2.set_title('Temporal Correlation vs Length Scale', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True)
    
    # Sigma sensitivity plots
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(results['sigma'], results['std_velocity'][len(L_values):], 'o-', color='blue', linewidth=4)
    ax3.set_xlabel('Intensity Scale (σ)', fontsize=20)
    ax3.set_ylabel('Standard Deviation (m/s)', fontsize=20)
    ax3.set_title('Sensitivity to Intensity Scale', fontsize=20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.grid(True)
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(results['sigma'], results['kurtosis'][len(L_values):], 'o-', color='blue', linewidth=4)
    ax4.set_xlabel('Intensity Scale (σ)', fontsize=20)
    ax4.set_ylabel('Kurtosis', fontsize=20)
    ax4.set_title('Distribution Shape vs Intensity Scale', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.grid(True)
    plt.savefig('results/verification/disturbance_generator_verification/sensitivity_analysis.png')
    
    # Print sensitivity metrics
    print("\n=== Sensitivity Analysis Results ===")
    print(f"Length Scale (L) Range: {min(L_values)} to {max(L_values)}")
    print(f"Intensity Scale (σ) Range: {min(sigma_values)} to {max(sigma_values)}")
    print("\nKey Metrics:")
    print(f"Mean Standard Deviation: {np.mean(results['std_velocity']):.3f}")
    print(f"Mean Kurtosis: {np.mean(results['kurtosis']):.3f}")
    print(f"Mean Autocorrelation: {np.mean(results['autocorr_lag1']):.3f}")
    print(f"Mean PSD Slope: {np.mean(results['psd_slope']):.3f}")
    
    # Calculate sensitivity coefficients
    L_sensitivity = np.polyfit(results['L'], results['std_velocity'][:len(L_values)], 1)[0]
    sigma_sensitivity = np.polyfit(results['sigma'], results['std_velocity'][len(L_values):], 1)[0]
    
    print("\nSensitivity Coefficients:")
    print(f"Length Scale Sensitivity: {L_sensitivity:.3f}")
    print(f"Intensity Scale Sensitivity: {sigma_sensitivity:.3f}")

    # Save key metrics to CSV
    key_metrics = {
        'Length Scale Min': min(L_values),
        'Length Scale Max': max(L_values),
        'Intensity Scale Min': min(sigma_values),
        'Intensity Scale Max': max(sigma_values),
        'Mean Standard Deviation': np.mean(results['std_velocity']),
        'Mean Kurtosis': np.mean(results['kurtosis']),
        'Mean Autocorrelation': np.mean(results['autocorr_lag1']),
        'Mean PSD Slope': np.mean(results['psd_slope'])
    }
    
    # Save key metrics to CSV
    results_df = pd.DataFrame(key_metrics, index=[0])
    results_df.to_csv('results/verification/disturbance_generator_verification/sensitivity_analysis.csv', index=False)
    print(f"Key metrics saved to 'results/verification/disturbance_generator_verification/sensitivity_analysis.csv'")

if __name__ == "__main__":
    analyze_parameter_sensitivity()