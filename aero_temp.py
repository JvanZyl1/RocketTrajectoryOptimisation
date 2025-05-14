import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

def load_coefficient_data(filename, coef_type="drag"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    aoa_values = []
    for val in header:
        if 'deg' in val:
            aoa_values.append(float(val.split('_')[0]))
    data = {angle: {'mach': [], 'coef': []} for angle in aoa_values}
    for line in lines[2:]:
        if not line.strip():
            continue
        values = line.strip().split(',')
        if len(values) < len(header):
            continue
        col_idx = 0
        for i, angle in enumerate(aoa_values):
            mach_idx = col_idx
            coef_idx = col_idx + 1
            if coef_idx < len(values) and values[mach_idx].strip() and values[coef_idx].strip():
                try:
                    mach_val = float(values[mach_idx])
                    coef_val = float(values[coef_idx])
                    data[angle]['mach'].append(mach_val)
                    data[angle]['coef'].append(coef_val)
                except ValueError:
                    pass
            col_idx += 2
    
    all_mach = []
    all_coef = []
    all_aoa = []
    
    for angle in aoa_values:
        mach_vals = data[angle]['mach']
        coef_vals = data[angle]['coef']
        
        all_mach.extend(mach_vals)
        all_coef.extend(coef_vals)
        all_aoa.extend([angle] * len(mach_vals))
    
    return np.array(all_mach), np.array(all_aoa), np.array(all_coef), np.array(aoa_values)

def load_drag_data(filename='V2_drag_coefficient.csv'):
    return load_coefficient_data(filename, coef_type="drag")

def load_lift_data(filename='V2_lift_coefficient.csv'):
    return load_coefficient_data(filename, coef_type="lift")

def create_coefficient_interpolator(mach, aoa, coef):
    points = np.column_stack((mach, aoa))
    interp = LinearNDInterpolator(points, coef)
    
    # Create fallback interpolator for extrapolation using nearest neighbor
    fallback = NearestNDInterpolator(points, coef)
    
    def interpolate_coef(mach_val, aoa_val):
        pts = np.array([[mach_val, aoa_val]])
        result = interp(pts)
        
        # If the result is NaN (outside convex hull), use nearest neighbor
        if np.isnan(result[0]):
            result[0] = fallback(pts)[0]
            
        return float(result[0])
    
    return interpolate_coef

def create_cd_interpolator(mach, aoa, cd):
    return create_coefficient_interpolator(mach, aoa, cd)

def create_cl_interpolator(mach, aoa, cl):
    return create_coefficient_interpolator(mach, aoa, cl)

def plot_coefficient_vs_mach_aoa(coef_interp_func, mach_range, aoa_values, coef_type="CD"):
    coef_labels = {
        "CD": r"$C_D$",
        "CL": r"$C_L$"
    }
    
    mach_grid = np.linspace(min(mach_range), max(mach_range), 50)
    plt.figure(figsize=(10, 5))
    for aoa in aoa_values:
        coef_values = np.array([coef_interp_func(m, aoa) for m in mach_grid])
        plt.plot(mach_grid, coef_values, label=rf'AoA = {math.degrees(aoa):.0f}$^\circ$', linewidth=4)
    plt.xlabel('Mach Number', fontsize=20)
    plt.ylabel(coef_labels.get(coef_type, coef_type), fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    # Save with appropriate filename
    if coef_type == "CD":
        plt.savefig('results/Sizing/drag_coefficient.png')
    elif coef_type == "CL":
        plt.savefig('results/Sizing/lift_coefficient.png')
    else:
        plt.savefig(f'results/Sizing/{coef_type.lower()}_coefficient.png')

def plot_cd_vs_mach_aoa(cd_interp_func, mach_range, aoa_values):
    return plot_coefficient_vs_mach_aoa(cd_interp_func, mach_range, aoa_values, coef_type="CD")

def plot_cl_vs_mach_aoa(cl_interp_func, mach_range, aoa_values):
    return plot_coefficient_vs_mach_aoa(cl_interp_func, mach_range, aoa_values, coef_type="CL")

def rocket_CD_compiler():
    mach, aoa, cd, aoa_values = load_drag_data()
    cd_interp = create_cd_interpolator(mach, aoa, cd)
    def fun(mach, aoa):
        return cd_interp(mach, abs(math.degrees(aoa)))
    return fun

def rocket_CL_compiler():
    """Compile lift coefficient interpolator"""
    mach, aoa, cl, aoa_values = load_lift_data()
    cl_interp = create_cl_interpolator(mach, aoa, cl)
    def fun(mach, aoa_radians): 
        return cl_interp(mach, abs(math.degrees(aoa_radians)))
    return fun

if __name__ == "__main__":
    # Process drag coefficient
    cd_interpolator = rocket_CD_compiler()
    plot_cd_vs_mach_aoa(cd_interpolator, np.linspace(0, 5, 100), np.deg2rad(np.array([2,4,6,8,10])))
    cl_interpolator = rocket_CL_compiler()
    plot_cl_vs_mach_aoa(cl_interpolator, np.linspace(0, 5, 100), np.deg2rad(np.array([2,4,6,8,10])))