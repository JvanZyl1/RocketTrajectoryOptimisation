import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Define Mach and angle-of-attack axes
mach_vals = np.array([2, 3, 5, 7, 10, 15, 20])
alpha_vals_deg = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
alpha_vals_rad = np.radians(alpha_vals_deg)
# file:///home/jonathanvanzyl/Downloads/T_1358249295Pezzellae-book.pdf : Annex 7
cl_data = np.array([
    [0.0404, 0.3044, 0.6696, 1.0821, 1.5215, 1.9753, 2.4140, 2.8051, 3.1353, 3.4052, 3.3238],
    [0.0292, 0.2458, 0.5472, 0.9007, 1.2903, 1.6959, 2.0922, 2.4519, 2.7505, 2.9646, 3.0917],
    [0.0196, 0.2015, 0.4612, 0.7791, 1.1439, 1.5328, 1.9191, 2.2730, 2.5701, 2.7844, 2.8956],
    [0.0157, 0.1838, 0.4284, 0.7371, 1.0967, 1.4836, 1.8698, 2.2268, 2.5258, 2.7420, 2.8560],
    [0.0131, 0.1720, 0.4077, 0.7126, 1.0711, 1.4588, 1.8468, 2.2060, 2.5077, 2.7259, 2.8397],
    [0.0115, 0.1642, 0.3958, 0.6997, 1.0591, 1.4485, 1.8384, 2.1989, 2.5022, 2.7205, 2.8349],
    [0.0110, 0.1614, 0.3917, 0.6959, 1.0560, 1.4463, 1.8368, 2.1978, 2.5016, 2.7197, 2.8342],
])
CL_interpolator = RegularGridInterpolator((mach_vals, alpha_vals_deg), cl_data)

def plot_CL():
    mach_fine = np.linspace(min(mach_vals), max(mach_vals), 100)
    alpha_fine = np.linspace(min(alpha_vals_deg), max(alpha_vals_deg), 100)
    mach_grid_fine, alpha_grid_fine = np.meshgrid(mach_fine, alpha_fine)
    points = np.vstack([mach_grid_fine.ravel(), alpha_grid_fine.ravel()]).T
    cl_fine = CL_interpolator(points).reshape(mach_grid_fine.shape)
    # Create a contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(alpha_grid_fine, mach_grid_fine, cl_fine, levels=20, cmap='viridis')
    plt.colorbar(contour, label=r'Lift Coefficient')
    contour_lines = plt.contour(alpha_grid_fine, mach_grid_fine, cl_fine, levels=10, colors='white', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=16, fmt='%.2f')

    plt.title('Supersonic lift coefficient')
    plt.xlabel(r'Angle of Attack ($^{\circ}$)', fontsize=18)
    plt.ylabel('Mach Number', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_CL()

