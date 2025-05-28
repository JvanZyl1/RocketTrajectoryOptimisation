import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats
import os

import sys
sys.path.append('.')
from src.envs.disturbance_generator import VKDisturbanceGenerator

def analyze_disturbance_generator_time_series():
    # Parameters
    dt = 0.02
    n_samples = 5000
    time = np.arange(0, n_samples * dt, dt)
    
    # Create generators with different parameters
    generators = {
        'Base Case': VKDisturbanceGenerator(dt=dt, V=100.0),
        'High V': VKDisturbanceGenerator(dt=dt, V=200.0),
        'Low V': VKDisturbanceGenerator(dt=dt, V=50.0)
    }
    
    # Generate time series
    u_series = {}
    v_series = {}
    
    for name, gen in generators.items():
        # Reset the generator to get fresh parameters
        gen.reset()
        print(f"\n{name}:")
        print(f"L_u: {gen.L_u:.1f} m, L_v: {gen.L_v:.1f} m")
        print(f"sigma_u: {gen.sigma_u:.3f}, sigma_v: {gen.sigma_v:.3f}")
        
        u_vals = []
        v_vals = []
        for _ in range(n_samples):
            u, v = gen()
            u_vals.append(u)
            v_vals.append(v)
        
        u_series[name] = np.array(u_vals)
        v_series[name] = np.array(v_vals)
    
    # Plot u-component time series
    plt.figure(figsize=(20, 15))
    plt.suptitle('VKDisturbanceGenerator Analysis - Horizontal Component', fontsize=24)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0, 0])
    for name, series in u_series.items():
        ax1.plot(time[:500], series[:500], label=name)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Horizontal Gust (m/s)', fontsize=20)
    ax1.set_title('Horizontal Gust Velocity Time Series', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)
    
    # Histogram
    ax2 = plt.subplot(gs[0, 1])
    for name, series in u_series.items():
        hist, bins = np.histogram(series, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(bin_centers, hist, label=name)
    ax2.set_xlabel('Horizontal Gust (m/s)', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.set_title('Horizontal Gust Distribution', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(fontsize=16)
    ax2.grid(True)
    
    # Autocorrelation
    ax3 = plt.subplot(gs[1, 0])
    for name, series in u_series.items():
        acf = np.correlate(series, series, mode='full') / (np.var(series) * len(series))
        ax3.plot(acf[len(acf)//2:len(acf)//2+100], label=name)
    ax3.set_xlabel('Lag', fontsize=20)
    ax3.set_ylabel('Autocorrelation', fontsize=20)
    ax3.set_title('Horizontal Gust Autocorrelation Function', fontsize=20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.legend(fontsize=16)
    ax3.grid(True)
    
    # Power Spectral Density
    ax4 = plt.subplot(gs[1, 1])
    for name, series in u_series.items():
        frequencies = np.fft.rfftfreq(len(series), dt)
        psd = np.abs(np.fft.rfft(series))**2 / len(series)
        ax4.loglog(frequencies[1:], psd[1:], label=name)
    ax4.set_xlabel('Frequency (Hz)', fontsize=20)
    ax4.set_ylabel('Power Spectral Density', fontsize=20)
    ax4.set_title('Horizontal Gust PSD', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.legend(fontsize=16)
    ax4.grid(True)
    
    os.makedirs('results/verification/disturbance_generator_verification', exist_ok=True)
    plt.savefig('results/verification/disturbance_generator_verification/vk_generator_u_analysis.png')
    plt.close()
    
    # Plot v-component time series
    plt.figure(figsize=(20, 15))
    plt.suptitle('VKDisturbanceGenerator Analysis - Vertical Component', fontsize=24)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0, 0])
    for name, series in v_series.items():
        ax1.plot(time[:500], series[:500], label=name)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Vertical Gust (m/s)', fontsize=20)
    ax1.set_title('Vertical Gust Velocity Time Series', fontsize=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)
    
    # Histogram
    ax2 = plt.subplot(gs[0, 1])
    for name, series in v_series.items():
        hist, bins = np.histogram(series, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax2.plot(bin_centers, hist, label=name)
    ax2.set_xlabel('Vertical Gust (m/s)', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.set_title('Vertical Gust Distribution', fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(fontsize=16)
    ax2.grid(True)
    
    # Autocorrelation
    ax3 = plt.subplot(gs[1, 0])
    for name, series in v_series.items():
        acf = np.correlate(series, series, mode='full') / (np.var(series) * len(series))
        ax3.plot(acf[len(acf)//2:len(acf)//2+100], label=name)
    ax3.set_xlabel('Lag', fontsize=20)
    ax3.set_ylabel('Autocorrelation', fontsize=20)
    ax3.set_title('Vertical Gust Autocorrelation Function', fontsize=20)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.legend(fontsize=16)
    ax3.grid(True)
    
    # Power Spectral Density
    ax4 = plt.subplot(gs[1, 1])
    for name, series in v_series.items():
        frequencies = np.fft.rfftfreq(len(series), dt)
        psd = np.abs(np.fft.rfft(series))**2 / len(series)
        ax4.loglog(frequencies[1:], psd[1:], label=name)
    ax4.set_xlabel('Frequency (Hz)', fontsize=20)
    ax4.set_ylabel('Power Spectral Density', fontsize=20)
    ax4.set_title('Vertical Gust PSD', fontsize=20)
    ax4.tick_params(axis='both', labelsize=16)
    ax4.legend(fontsize=16)
    ax4.grid(True)
    
    plt.savefig('results/verification/disturbance_generator_verification/vk_generator_v_analysis.png')
    plt.close()
    
    # Calculate statistics and save to CSV
    results = []
    for name in generators.keys():
        u_stats = {
            'Name': f"{name} (U)",
            'Mean': np.mean(u_series[name]),
            'Std Dev': np.std(u_series[name]),
            'Min': np.min(u_series[name]),
            'Max': np.max(u_series[name]),
            'Skewness': stats.skew(u_series[name]),
            'Kurtosis': stats.kurtosis(u_series[name])
        }
        results.append(u_stats)
        
        v_stats = {
            'Name': f"{name} (V)",
            'Mean': np.mean(v_series[name]),
            'Std Dev': np.std(v_series[name]),
            'Min': np.min(v_series[name]),
            'Max': np.max(v_series[name]),
            'Skewness': stats.skew(v_series[name]),
            'Kurtosis': stats.kurtosis(v_series[name])
        }
        results.append(v_stats)
    
    # Save key metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/verification/disturbance_generator_verification/vk_generator_statistics.csv', index=False)
    print(f"Analysis complete. Results saved to 'results/verification/disturbance_generator_verification/'")

if __name__ == "__main__":
    analyze_disturbance_generator_time_series() 