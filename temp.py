# Re-imports and function redefinitions for clean state
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Redefine the original piecewise Saturn V CD(M) function
def piecewise_CD(M):
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
    elif M <= 2.0:
        return -0.04373178 * M**3 + 0.3236152 * M**2 - 1.019679 * M + 1.554752
    elif M <= 3.25:
        return 0.01024 * M**3 - 0.00864 * M**2 - 0.33832 * M + 1.08928
    elif M <= 4.5:
        return -0.01408 * M**3 + 0.19168 * M**2 - 0.86976 * M + 1.53544
    else:
        return 0.22

# Recompute values
M_vals = np.linspace(0.1, 5.0, 900)
CD_vals = np.array([piecewise_CD(M) for M in M_vals])


# Fit a polynomial (degree 5 for smoothness) to the subset
def poly_fit_sub(M, a0, a1, a2, a3, a4, a5):
    return a0 + a1*M + a2*M**2 + a3*M**3 + a4*M**4 + a5*M**5

params_sup, _ = curve_fit(poly_fit_sub, M_vals[(M_vals >= 1.15) & (M_vals <=4.5)], CD_vals[(M_vals >= 1.15) & (M_vals <=4.5)])
CD_fit_sup = poly_fit_sub(M_vals[(M_vals >= 1.15) & (M_vals <=4.5)], *params_sup)

params_sub, _ = curve_fit(poly_fit_sub, M_vals[(M_vals >= 0.8) & (M_vals <=1.15)], CD_vals[(M_vals >= 0.8) & (M_vals <=1.15)])
CD_fit_sub = poly_fit_sub(M_vals[(M_vals >= 0.8) & (M_vals <=1.15)], *params_sub)

C_D_0 = CD_fit_sub[0]
C_D_0_vals = np.array([C_D_0 for M in M_vals[(M_vals <= 0.8)]])

# Plot subset and fit
plt.figure(figsize=(10, 5))
plt.plot(M_vals, CD_vals, label="Original Saturn V $C_D(M)$", linewidth=2)
plt.plot(M_vals[(M_vals >= 1.15) & (M_vals <=4.5)], CD_fit_sup, '--', label="Polynomial Fit (deg 5)", linewidth=2)
plt.plot(M_vals[(M_vals >= 0.8) & (M_vals <=1.15)], CD_fit_sub, '--', label="Polynomial Fit (deg 5)", linewidth=2)
plt.plot(M_vals[(M_vals <= 0.8)], C_D_0_vals, '--', label="Polynomial Fit (deg 5)", linewidth=2)
plt.xlabel("Mach number $M$")
plt.ylabel("Drag coefficient $C_D$")
plt.title("Polynomial Fit to Saturn V Drag Coefficient (1.15 ≤ M ≤ 5.0, α = 0°)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()