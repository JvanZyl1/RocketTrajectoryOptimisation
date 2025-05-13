import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

def rocket_CL(alpha, M, C_L_0, C_L_alpha_sub):
    if M < 1.0:
        return C_L_0 + C_L_alpha_sub * alpha
    elif M <= 1.2:
        lambda_val = (M - 0.8) / 0.4
        return (1 - lambda_val) * (C_L_0 + C_L_alpha_sub * alpha) + lambda_val * 4 * alpha / math.sqrt(M**2 -1)
    else:
        return 4 * alpha / math.sqrt(M**2 -1)

def rocket_CD(alpha, M, C_D_0, k, C_L_0, C_L_alpha_sub, m_fac, delta_C_D):
    if M < 0.8:
        C_L_val = C_L_0 + C_L_alpha_sub * alpha
        return C_D_0 + k * C_L_val**2
    elif M <= 1.0:
        return C_D_0 + delta_C_D * ((M - 0.8)/0.2)**2
    elif M <= 1.2:
        return C_D_0 + delta_C_D * (2 - M)
    else:
        return 4*(alpha**2 + m_fac)/math.sqrt(M**2 -1)
    


# Parameters
# C_L_0 : zero angle of attack lift coefficient
# C_L_alpha_sub : subsonic lift coefficient angle of attack slope
# k := 1/(pi * e * AR)
# m_fac : supersonic non symmetrical airfoil contribution

def plot_CL_variation():
    # Vary C_L_alpha_sub
    C_L_alpha_vals = (4.5, 6.5)
    C_L_0_vals = (0.0, 0.1)
    C_D_0_vals = (0.02, 0.1)
    k_vals = (0.1, 0.3)
    m_fac_vals = (0.0, (5 * math.pi/180)**2)
    delta_C_D_vals = (0.3, 1.2)

    # Mach and alpha range
    M_range = np.linspace(0.1, 2.0, 100)
    alpha = math.radians(5.0)

    # Initialize arrays to store min/max values
    CL_min = np.ones_like(M_range) * float('inf')
    CL_max = np.ones_like(M_range) * float('-inf')
    CD_min = np.ones_like(M_range) * float('inf')
    CD_max = np.ones_like(M_range) * float('-inf')

    # Across the M_range, try every combination of the parameters
    for C_L_alpha_sub in C_L_alpha_vals:
        for C_L_0 in C_L_0_vals:
            for C_D_0 in C_D_0_vals:
                for k in k_vals:
                    for m_fac in m_fac_vals:
                        for delta_C_D in delta_C_D_vals:
                            CL_values = np.array([rocket_CL(alpha, M, C_L_0, C_L_alpha_sub) for M in M_range])
                            CD_values = np.array([rocket_CD(alpha, M, C_D_0, k, C_L_0, C_L_alpha_sub, m_fac, delta_C_D) for M in M_range])
                            
                            # Update min/max values
                            CL_min = np.minimum(CL_min, CL_values)
                            CL_max = np.maximum(CL_max, CL_values)
                            CD_min = np.minimum(CD_min, CD_values)
                            CD_max = np.maximum(CD_max, CD_values)
    # Default parameters
    C_L_0 = 0.0
    C_L_alpha_sub = 5.5
    C_D_0 = (0.1-0.02)/2
    k = (0.53-0.1)/2
    m_fac = (5 * math.pi/180)**2/2
    delta_C_D = 0.7
    C_D_vals_nom = np.array([rocket_CD(alpha, M, C_D_0, k, C_L_0, C_L_alpha_sub, m_fac, delta_C_D) for M in M_range])
    C_L_vals_nom = np.array([rocket_CL(alpha, M, C_L_0, C_L_alpha_sub) for M in M_range])


    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Plot CL
    ax1 = plt.subplot(gs[0])
    ax1.fill_between(M_range, CL_min, CL_max, alpha=0.3, color='blue', label='CL Range')
    ax1.plot(M_range, C_L_vals_nom, 'b-', label='Nominal')
    ax1.set_ylabel('Lift Coefficient ($C_L$)')
    ax1.set_title(f'Lift Coefficient vs Mach Number ($\\alpha$ = {math.degrees(alpha):.1f}$^{{\\circ}}$)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot CD
    ax2 = plt.subplot(gs[1])
    ax2.fill_between(M_range, CD_min, CD_max, alpha=0.3, color='red', label='CD Range')
    ax2.plot(M_range, C_D_vals_nom, 'r-', label='Nominal')
    ax2.set_xlabel('Mach Number')
    ax2.set_ylabel('Drag Coefficient ($C_D$)')
    ax2.set_title(f'Drag Coefficient vs Mach Number ($\\alpha$ = {math.degrees(alpha):.1f}$^{{\\circ}}$)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_CD_CL_vs_alpha(M_values=[0.5, 0.9, 1.1, 2.0, 4.0]):
    """Plot drag coefficient variation with angle of attack at different Mach numbers"""
    # Default parameters
    C_L_0 = 2 * math.pi
    C_L_alpha_sub = math.pi/180
    C_D_0 = (0.1-0.02)/2
    k = (0.53-0.1)/2
    m_fac = (5 * math.pi/180)**2/2
    delta_C_D = 5.5
    
    # Alpha range in degrees
    alpha_deg = np.linspace(-10, 10, 100)
    alpha_rad = np.radians(alpha_deg)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot CD
    for M in M_values:
        CD_values = np.array([rocket_CD(alpha, M, C_D_0, k, C_L_0, C_L_alpha_sub, m_fac, delta_C_D) for alpha in alpha_rad])
        ax1.plot(alpha_deg, CD_values, label=f'M = {M}')
    
    ax1.set_ylabel('Drag Coefficient ($C_D$)')
    ax1.set_title('Drag Coefficient vs Angle of Attack')
    ax1.grid(True)
    ax1.legend()
    
    # Plot CL
    for M in M_values:
        CL_values = np.array([rocket_CL(alpha, M, C_L_0, C_L_alpha_sub) for alpha in alpha_rad])
        ax2.plot(alpha_deg, CL_values, label=f'M = {M}')
    
    ax2.set_xlabel(r'$\alpha$ ($^{\circ}$)')
    ax2.set_ylabel('Lift Coefficient ($C_L$)')
    ax2.set_title('Lift Coefficient vs Angle of Attack')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    plot_CL_variation()
    #plot_CD_CL_vs_alpha()