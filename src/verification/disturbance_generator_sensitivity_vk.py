import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import pandas as pd
import os

import sys
sys.path.append('.')
from src.envs.disturbance_generator import VKDisturbanceGenerator

def analyze_vk_generator_sensitivity():
    """Analyze the sensitivity of VKDisturbanceGenerator to changes in velocity parameter V."""
    # Parameters
    dt = 0.02
    n_samples = 5000
    n_runs = 10  # Number of runs per V value to account for random parameter sampling
    
    # V values to test
    V_values = np.linspace(50, 200, 6)
    
    # Store results
    results = {
        'V': [],
        'L_u': [], 'L_v': [],
        'sigma_u': [], 'sigma_v': [],
        'u_std': [], 'v_std': [],
        'u_acf1': [], 'v_acf1': [],
        'u_spectral_width': [], 'v_spectral_width': []
    }
    
    print("Running sensitivity analysis for VKDisturbanceGenerator...")
    print(f"Testing {len(V_values)} different V values with {n_runs} runs each")
    
    # For each V value, run multiple times with different random parameters
    for V in V_values:
        print(f"\nV = {V:.1f} m/s:")
        
        for run in range(n_runs):
            # Create generator and reset to get fresh parameters
            gen = VKDisturbanceGenerator(dt=dt, V=V)
            gen.reset()
            
            # Record parameters
            results['V'].append(V)
            results['L_u'].append(gen.L_u)
            results['L_v'].append(gen.L_v)
            results['sigma_u'].append(gen.sigma_u)
            results['sigma_v'].append(gen.sigma_v)
            
            # Generate time series
            u_vals = []
            v_vals = []
            for _ in range(n_samples):
                u, v = gen()
                u_vals.append(u)
                v_vals.append(v)
            
            u_series = np.array(u_vals)
            v_series = np.array(v_vals)
            
            # Calculate metrics
            # 1. Standard deviation
            results['u_std'].append(np.std(u_series))
            results['v_std'].append(np.std(v_series))
            
            # 2. Lag-1 autocorrelation
            results['u_acf1'].append(np.corrcoef(u_series[:-1], u_series[1:])[0,1])
            results['v_acf1'].append(np.corrcoef(v_series[:-1], v_series[1:])[0,1])
            
            # 3. Spectral width (using FFT)
            u_frequencies = np.fft.rfftfreq(len(u_series), dt)
            u_psd = np.abs(np.fft.rfft(u_series))**2 / len(u_series)
            u_psd_normalized = u_psd / np.sum(u_psd)
            u_spectral_centroid = np.sum(u_frequencies * u_psd_normalized)
            u_spectral_width = np.sqrt(np.sum((u_frequencies - u_spectral_centroid)**2 * u_psd_normalized))
            results['u_spectral_width'].append(u_spectral_width)
            
            v_frequencies = np.fft.rfftfreq(len(v_series), dt)
            v_psd = np.abs(np.fft.rfft(v_series))**2 / len(v_series)
            v_psd_normalized = v_psd / np.sum(v_psd)
            v_spectral_centroid = np.sum(v_frequencies * v_psd_normalized)
            v_spectral_width = np.sqrt(np.sum((v_frequencies - v_spectral_centroid)**2 * v_psd_normalized))
            results['v_spectral_width'].append(v_spectral_width)
            
            print(f"  Run {run+1}: L_u={gen.L_u:.1f}m, L_v={gen.L_v:.1f}m, σ_u={gen.sigma_u:.2f}, σ_v={gen.sigma_v:.2f}")
    
    # Convert to pandas DataFrame for easy manipulation
    df = pd.DataFrame(results)
    
    # Create output directory
    os.makedirs('results/verification/disturbance_generator_verification', exist_ok=True)
    
    # Group by V and calculate mean and std for each metric
    grouped = df.groupby('V')
    agg_results = grouped.agg({
        'u_std': ['mean', 'std'],
        'v_std': ['mean', 'std'],
        'u_acf1': ['mean', 'std'],
        'v_acf1': ['mean', 'std'],
        'u_spectral_width': ['mean', 'std'],
        'v_spectral_width': ['mean', 'std']
    })
    
    # Plot results
    plt.figure(figsize=(20, 15))
    plt.suptitle('VKDisturbanceGenerator Sensitivity to V Parameter', fontsize=24)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    # Plot 1: Standard Deviation vs V
    ax1 = plt.subplot(gs[0, 0])
    V_unique = sorted(df['V'].unique())
    u_std_mean = [agg_results.loc[v, ('u_std', 'mean')] for v in V_unique]
    u_std_std = [agg_results.loc[v, ('u_std', 'std')] for v in V_unique]
    v_std_mean = [agg_results.loc[v, ('v_std', 'mean')] for v in V_unique]
    v_std_std = [agg_results.loc[v, ('v_std', 'std')] for v in V_unique]
    
    ax1.errorbar(V_unique, u_std_mean, yerr=u_std_std, fmt='o-', label='Horizontal (u)', capsize=5, linewidth=2)
    ax1.errorbar(V_unique, v_std_mean, yerr=v_std_std, fmt='s-', label='Vertical (v)', capsize=5, linewidth=2)
    ax1.set_xlabel('V (m/s)', fontsize=16)
    ax1.set_ylabel('Standard Deviation (m/s)', fontsize=16)
    ax1.set_title('Gust Intensity vs Velocity', fontsize=18)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=14)
    
    # Plot 2: Autocorrelation vs V
    ax2 = plt.subplot(gs[0, 1])
    u_acf1_mean = [agg_results.loc[v, ('u_acf1', 'mean')] for v in V_unique]
    u_acf1_std = [agg_results.loc[v, ('u_acf1', 'std')] for v in V_unique]
    v_acf1_mean = [agg_results.loc[v, ('v_acf1', 'mean')] for v in V_unique]
    v_acf1_std = [agg_results.loc[v, ('v_acf1', 'std')] for v in V_unique]
    
    ax2.errorbar(V_unique, u_acf1_mean, yerr=u_acf1_std, fmt='o-', label='Horizontal (u)', capsize=5, linewidth=2)
    ax2.errorbar(V_unique, v_acf1_mean, yerr=v_acf1_std, fmt='s-', label='Vertical (v)', capsize=5, linewidth=2)
    ax2.set_xlabel('V (m/s)', fontsize=16)
    ax2.set_ylabel('Lag-1 Autocorrelation', fontsize=16)
    ax2.set_title('Correlation vs Velocity', fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=14)
    
    # Plot 3: Spectral Width vs V
    ax3 = plt.subplot(gs[0, 2])
    u_sw_mean = [agg_results.loc[v, ('u_spectral_width', 'mean')] for v in V_unique]
    u_sw_std = [agg_results.loc[v, ('u_spectral_width', 'std')] for v in V_unique]
    v_sw_mean = [agg_results.loc[v, ('v_spectral_width', 'mean')] for v in V_unique]
    v_sw_std = [agg_results.loc[v, ('v_spectral_width', 'std')] for v in V_unique]
    
    ax3.errorbar(V_unique, u_sw_mean, yerr=u_sw_std, fmt='o-', label='Horizontal (u)', capsize=5, linewidth=2)
    ax3.errorbar(V_unique, v_sw_mean, yerr=v_sw_std, fmt='s-', label='Vertical (v)', capsize=5, linewidth=2)
    ax3.set_xlabel('V (m/s)', fontsize=16)
    ax3.set_ylabel('Spectral Width (Hz)', fontsize=16)
    ax3.set_title('Frequency Content vs Velocity', fontsize=18)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True)
    ax3.legend(fontsize=14)
    
    # Plot 4: Scatter plot of L_u vs V
    ax4 = plt.subplot(gs[1, 0])
    ax4.scatter(df['V'], df['L_u'], alpha=0.7, label='L_u')
    ax4.scatter(df['V'], df['L_v'], alpha=0.7, label='L_v')
    ax4.set_xlabel('V (m/s)', fontsize=16)
    ax4.set_ylabel('Length Scale (m)', fontsize=16)
    ax4.set_title('Length Scale Sampling vs Velocity', fontsize=18)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.grid(True)
    ax4.legend(fontsize=14)
    
    # Plot 5: Scatter plot of sigma_u and sigma_v vs V
    ax5 = plt.subplot(gs[1, 1])
    ax5.scatter(df['V'], df['sigma_u'], alpha=0.7, label='sigma_u')
    ax5.scatter(df['V'], df['sigma_v'], alpha=0.7, label='sigma_v')
    ax5.set_xlabel('V (m/s)', fontsize=16)
    ax5.set_ylabel('Sigma', fontsize=16)
    ax5.set_title('Intensity Parameter Sampling vs Velocity', fontsize=18)
    ax5.tick_params(axis='both', labelsize=14)
    ax5.grid(True)
    ax5.legend(fontsize=14)
    
    # Plot 6: Theoretical vs Measured Relationship
    ax6 = plt.subplot(gs[1, 2])
    # Compute expected autocorrelation based on theory: exp(-V*dt/L)
    expected_u_acf = [np.mean([np.exp(-v*dt/l) for l in df[df['V']==v]['L_u']]) for v in V_unique]
    expected_v_acf = [np.mean([np.exp(-v*dt/l) for l in df[df['V']==v]['L_v']]) for v in V_unique]
    
    ax6.plot(V_unique, expected_u_acf, 'o--', label='Expected u (theory)', linewidth=2)
    ax6.plot(V_unique, expected_v_acf, 's--', label='Expected v (theory)', linewidth=2)
    ax6.plot(V_unique, u_acf1_mean, 'o-', label='Measured u', linewidth=2)
    ax6.plot(V_unique, v_acf1_mean, 's-', label='Measured v', linewidth=2)
    ax6.set_xlabel('V (m/s)', fontsize=16)
    ax6.set_ylabel('Lag-1 Autocorrelation', fontsize=16)
    ax6.set_title('Theory vs Measurement', fontsize=18)
    ax6.tick_params(axis='both', labelsize=14)
    ax6.grid(True)
    ax6.legend(fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/verification/disturbance_generator_verification/vk_generator_sensitivity.png')
    
    # Save results to CSV
    df.to_csv('results/verification/disturbance_generator_verification/vk_generator_sensitivity_raw.csv', index=False)
    agg_results.to_csv('results/verification/disturbance_generator_verification/vk_generator_sensitivity_summary.csv')
    
    print("\nSensitivity Analysis Results:")
    print("============================")
    print(f"Velocity (V) values tested: {V_unique}")
    print("\nEffect on autocorrelation:")
    for i, v in enumerate(V_unique):
        print(f"  V = {v:.1f} m/s: ACF1_u = {u_acf1_mean[i]:.4f} ± {u_acf1_std[i]:.4f}, ACF1_v = {v_acf1_mean[i]:.4f} ± {v_acf1_std[i]:.4f}")
    
    print("\nEffect on standard deviation:")
    for i, v in enumerate(V_unique):
        print(f"  V = {v:.1f} m/s: STD_u = {u_std_mean[i]:.4f} ± {u_std_std[i]:.4f}, STD_v = {v_std_mean[i]:.4f} ± {v_std_std[i]:.4f}")
    
    print("\nEffect on spectral width:")
    for i, v in enumerate(V_unique):
        print(f"  V = {v:.1f} m/s: SW_u = {u_sw_mean[i]:.4f} ± {u_sw_std[i]:.4f}, SW_v = {v_sw_mean[i]:.4f} ± {v_sw_std[i]:.4f}")
    
    print("\nAnalysis complete. Results saved to 'results/verification/disturbance_generator_verification/'")

if __name__ == "__main__":
    analyze_vk_generator_sensitivity() 