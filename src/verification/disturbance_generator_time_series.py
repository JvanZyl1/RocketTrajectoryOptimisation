import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats
import seaborn as sns

import sys
sys.path.append('.')
from src.envs.disturbance_generator import VonKarmanFilter

def analyze_time_series():
    # Parameters
    dt = 0.02
    n_samples = 10000
    time = np.arange(0, n_samples * dt, dt)
    
    # Create filters with different parameters
    filters = {
        'Base Case': VonKarmanFilter(L=200.0, sigma=1.0, V=100.0, dt=dt),
        'High L': VonKarmanFilter(L=500.0, sigma=1.0, V=100.0, dt=dt),
        'Low L': VonKarmanFilter(L=50.0, sigma=1.0, V=100.0, dt=dt),
        'High Sigma': VonKarmanFilter(L=200.0, sigma=2.0, V=100.0, dt=dt),
        'Low Sigma': VonKarmanFilter(L=200.0, sigma=0.5, V=100.0, dt=dt)
    }
    
    # Generate time series
    time_series = {}
    for name, filter in filters.items():
        time_series[name] = np.array([filter.step() for _ in range(n_samples)])
    
    # Plot time series
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])
    for name, series in time_series.items():
        ax1.plot(time[:1000], series[:1000], label=name)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Gust Velocity (m/s)', fontsize=20)
    ax1.set_title('Gust Velocity Time Series', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)
    
    # Histogram
    ax2 = plt.subplot(gs[0, 1])
    for name, series in time_series.items():
        sns.kdeplot(data=series, ax=ax2, label=name)
    ax2.set_xlabel('Gust Velocity (m/s)', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.set_title('Velocity Distribution', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(fontsize=16)
    ax2.grid(True)
    # Autocorrelation
    ax3 = plt.subplot(gs[1, 0])
    for name, series in time_series.items():
        acf = np.correlate(series, series, mode='full') / (np.var(series) * len(series))
        ax3.plot(acf[len(acf)//2:len(acf)//2+100], label=name)
    ax3.set_xlabel('Lag', fontsize=20)
    ax3.set_ylabel('Autocorrelation', fontsize=20)
    ax3.set_title('Autocorrelation Function', fontsize=20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.legend(fontsize=16)
    ax3.grid(True)
    # Power Spectral Density
    ax4 = plt.subplot(gs[1, 1])
    for name, series in time_series.items():
        f_psd, Pxx = plt.psd(series, Fs=1/dt, NFFT=1024, detrend='linear')
        ax4.loglog(f_psd, Pxx, label=name)
    ax4.set_xlabel('Frequency (Hz)', fontsize=20)
    ax4.set_ylabel('Power Spectral Density', fontsize=20)
    ax4.set_title('Power Spectral Density', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.legend(fontsize=16)
    plt.savefig('results/verification/disturbance_generator_verification/time_series_analysis.png')
    
    # Calculate and print statistics
    print("\n=== Time Series Analysis Results ===")
    key_metrics = {}
    for name, series in time_series.items():
        print(f"\n{name}:")
        print(f"Mean: {np.mean(series):.3f} m/s")
        print(f"Standard Deviation: {np.std(series):.3f} m/s")
        print(f"Skewness: {stats.skew(series):.3f}")
        print(f"Kurtosis: {stats.kurtosis(series):.3f}")
        
        # Calculate characteristic time scale
        acf = np.correlate(series, series, mode='full') / (np.var(series) * len(series))
        acf = acf[len(acf)//2:]
        tau = np.argmax(acf < 1/np.e) * dt
        print(f"Characteristic Time Scale: {tau:.3f} s")
        
        # Calculate energy content
        f_psd, Pxx = plt.psd(series, Fs=1/dt, NFFT=1024, detrend='linear')
        energy = np.trapz(Pxx, f_psd)
        print(f"Total Energy: {energy:.3f} (m/s)²")

        key_metrics[name] = {
            'Mean': np.mean(series),
            'Standard Deviation': np.std(series),
            'Skewness': stats.skew(series),
            'Kurtosis': stats.kurtosis(series),
            'Characteristic Time Scale': tau
        }

    # Save key metrics to CSV
    key_metrics_df = pd.DataFrame(key_metrics)
    key_metrics_df.to_csv('results/verification/disturbance_generator_verification/time_series_analysis.csv', index=False)

if __name__ == "__main__":
    analyze_time_series() 