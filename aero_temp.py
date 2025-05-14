import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator


def load_drag_data(filename='V2_drag_coefficient.csv'):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    aoa_values = []
    for val in header:
        if 'deg' in val:
            aoa_values.append(float(val.split('_')[0]))
    data = {angle: {'mach': [], 'cd': []} for angle in aoa_values}
    for line in lines[2:]:
        if not line.strip():
            continue
        values = line.strip().split(',')
        if len(values) < len(header):
            continue
        col_idx = 0
        for i, angle in enumerate(aoa_values):
            mach_idx = col_idx
            cd_idx = col_idx + 1
            if cd_idx < len(values) and values[mach_idx].strip() and values[cd_idx].strip():
                try:
                    mach_val = float(values[mach_idx])
                    cd_val = float(values[cd_idx])
                    data[angle]['mach'].append(mach_val)
                    data[angle]['cd'].append(cd_val)
                except ValueError:
                    pass
            col_idx += 2
    
    all_mach = []
    all_cd = []
    all_aoa = []
    for angle in aoa_values:
        mach_vals = data[angle]['mach']
        cd_vals = data[angle]['cd']
        
        all_mach.extend(mach_vals)
        all_cd.extend(cd_vals)
        all_aoa.extend([angle] * len(mach_vals))
    
    return np.array(all_mach), np.array(all_aoa), np.array(all_cd), np.array(aoa_values)

def create_cd_interpolator(mach, aoa, cd):
    points = np.column_stack((mach, aoa))
    interp = LinearNDInterpolator(points, cd)
    
    # Create fallback interpolator for extrapolation using nearest neighbor
    fallback = NearestNDInterpolator(points, cd)
    
    def interpolate_cd(mach_val, aoa_val):
        pts = np.array([[mach_val, aoa_val]])
        result = interp(pts)
        
        # If the result is NaN (outside convex hull), use nearest neighbor
        if np.isnan(result[0]):
            result[0] = fallback(pts)[0]
            
        return float(result[0])
    
    return interpolate_cd


def plot_cd_vs_mach_aoa(cd_interp_func, mach_range, aoa_values):
    mach_grid = np.linspace(min(mach_range), max(mach_range), 50)
    plt.figure(figsize=(10, 5))
    for aoa in aoa_values:
        cd_values = np.array([cd_interp_func(m, aoa) for m in mach_grid])
        plt.plot(mach_grid, cd_values, label=rf'AoA = {aoa}$^\circ$', linewidth = 4)
    plt.xlabel('Mach Number', fontsize = 20)
    plt.ylabel(r'$C_D$', fontsize = 20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize = 20)
    plt.tight_layout()
    plt.savefig('results/Sizing/drag_coefficient.png')

def rocket_CD_compiler():
    mach, aoa, cd, aoa_values = load_drag_data()
    cd_interp = create_cd_interpolator(mach, aoa, cd)
    def fun(mach, aoa):
        return cd_interp(mach, abs(aoa))
    return fun

if __name__ == "__main__":
    cd_interpolator = rocket_CD_compiler()
    plot_cd_vs_mach_aoa(cd_interpolator, np.linspace(0, 5, 100), np.linspace(0, 10, 5))
