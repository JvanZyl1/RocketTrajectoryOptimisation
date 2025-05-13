import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

def rocket_CL(alpha,                    # [rad]
              M,                        # [-]
              kl_sub = 2.0,             # effective lift slope in subsonic flight for a typical rocket
              kl_sup = 1.0              # reduced lift slope in supersonic flight
              ): # radians & -
    """
    For a rocket, the overall normal force coefficient derivative is lower than the thin-airfoil value.
    Here we assume:
      - Subsonic (M < 0.8): effective lift slope circa 2.0 per radian, with Prandtl-Glauert compressibility correction.
      - Transonic (0.8 leq M geq 1.2): linear interpolation between subsonic and supersonic slopes.
      - Supersonic (M > 1.2): reduced lift slope circa 1.0 per radian.
    """
    
    if M < 0.8:
        comp_factor = 1.0 / math.sqrt(1 - M**2)
        return kl_sub * alpha * comp_factor
    elif M <= 1.2:
        t = (M - 0.8) / 0.4
        # Evaluate subsonic value at M = 0.8
        comp_sub = 1.0 / math.sqrt(1 - 0.8**2)
        sub_val = kl_sub * alpha * comp_sub
        sup_val = kl_sup * alpha
        return (1 - t) * sub_val + t * sup_val
    else:
        return kl_sup * alpha

def rocket_CD(M):
    if M <= 0.6:
        return 0.2083333 * M**2 - 0.25 * M + 0.46
    elif M <= 0.8:
        return 1.25 * M**3 - 2.125 * M**2 + 1.2 * M + 0.16
    elif M <= 0.95:
        return 10.37037 * M**3 - 22.88889 * M**2 + 16.91111 * M - 3.78963
    elif M <= 1.05:
        return -30 * M**3 + 88.5 * M**2 - 85.425 * M + 27.51375
    elif M <= 1.15:
        return -20 * M**3 + 60 * M**2 - 58.65 * M + 19.245
    elif M <= 1.3:
        return 11.85185 * M**3 - 44.88889 * M**2 + 56.22222 * M - 22.58519
    elif M <= 2.0:
        return -0.04373178 * M**3 + 0.3236152 * M**2 - 1.019679 * M + 1.554752
    elif M <= 3.25:
        return 0.01024 * M**3 - 0.00864 * M**2 - 0.33832 * M + 1.08928
    elif M <= 4.5:
        return -0.01408 * M**3 + 0.19168 * M**2 - 0.86976 * M + 1.53544
    else:
        return 0.22

def plot_CD():
    mach = np.linspace(0.0, 5.0, 100)
    CD = [rocket_CD(M) for M in mach]
    plt.figure(figsize=(10, 5))
    plt.suptitle(r'Drag coefficient', fontsize=24)
    plt.plot(mach, CD, label='CD', linewidth=4, color='blue')
    plt.xlabel('Mach number', fontsize=18)
    plt.ylabel(r'$C_D$', fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('results/Sizing/CD_with_M.png')
    plt.close()

if __name__ == "__main__":
    plot_CD()