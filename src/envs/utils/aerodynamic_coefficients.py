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

def rocket_CD(alpha,                # [rad]
              M,                    # [-]
              cd0_subsonic=0.05,    # zero-lift drag coefficient in subsonic flight
              kd_subsonic=0.5,      # induced drag scaling in subsonic flight
              cd0_supersonic=0.10, # zero-lift drag coefficient in supersonic flight
              kd_supersonic=1.0    # induced drag scaling in supersonic flight
              ):
    """
    For a rocket, the drag is composed of:
      - A baseline zero-lift drag (cd0) that accounts for body, fin, and wave drag effects.
      - An induced drag term that scales roughly as α².
    We assume:
      - Subsonic (M < 0.8): cd0_subsonic circa 0.05 (with compressibility correction) and induced drag scaling kd_subsonic circa 0.5.
      - Transonic (0.8 leq M geq 1.2): linear interpolation between subsonic and supersonic parameters.
      - Supersonic (M > 1.2): cd0_supersonic circa 0.10 and induced drag scaling kd_supersonic circa 1.0.
    """
    if M < 0.8:
        comp_factor = 1.0 / math.sqrt(1 - M**2)
        return cd0_subsonic * comp_factor + kd_subsonic * (alpha**2)
    elif M <= 1.2:
        t = (M - 0.8) / 0.4
        comp_sub = 1.0 / math.sqrt(1 - 0.8**2)
        sub_val = cd0_subsonic * comp_sub + kd_subsonic * (alpha**2)
        sup_val = cd0_supersonic + kd_supersonic * (alpha**2)
        return (1 - t) * sub_val + t * sup_val
    else:
        return cd0_supersonic + kd_supersonic * (alpha**2)
    
def compile_drag_coefficient_func(alpha_degrees):
    return lambda M: rocket_CD(math.radians(alpha_degrees), M)

def plot_sensitivity_analysis(machs, alphas, kl_sub_range, kl_sup_range):
    plt.figure(figsize=(20, 10))
    plt.suptitle('Sensitivity Analysis of $C_L$ with Varying $k_{l_{sub}}$ and $k_{l_{sup}}$', fontsize=24)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Subsonic plot
    ax1 = plt.subplot(gs[0])
    for kl_sub in kl_sub_range:
        C_Ls_sub = [rocket_CL(alpha, 0.5, kl_sub=kl_sub) for alpha in alphas]
        ax1.plot(np.rad2deg(alphas), C_Ls_sub, label=f'$k_{{l_{{sub}}}}={kl_sub}$', linewidth=2)
    ax1.set_title('Subsonic (M = 0.5)', fontsize=22)
    ax1.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax1.set_ylabel(r'$C_L$', fontsize=20)
    ax1.legend(fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # Supersonic plot
    ax2 = plt.subplot(gs[1])
    for kl_sup in kl_sup_range:
        C_Ls_sup = [rocket_CL(alpha, 2.0, kl_sup=kl_sup) for alpha in alphas]
        ax2.plot(np.rad2deg(alphas), C_Ls_sup, label=f'$k_{{l_{{sup}}}}={kl_sup}$', linewidth=2)
    ax2.set_title('Supersonic (M = 2.0)', fontsize=22)
    ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax2.set_ylabel(r'$C_L$', fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('results/Sizing/CL_sensitivity_analysis.png')
    plt.close()

def plot_drag_sensitivity_analysis(alphas, cd0_sub_range, kd_sub_range, cd0_sup_range, kd_sup_range):
    plt.figure(figsize=(20, 15))
    plt.suptitle('Sensitivity Analysis of $C_D$', fontsize=24)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Subsonic plots
    ax1 = plt.subplot(gs[0, 0])
    for cd0_sub in cd0_sub_range:
        C_Ds_sub = [rocket_CD(alpha, 0.5, cd0_subsonic=cd0_sub) for alpha in alphas]
        ax1.plot(np.rad2deg(alphas), C_Ds_sub, label=f'$cd0_{{sub}}={cd0_sub}$', linewidth=2)
    ax1.set_title('Subsonic (M = 0.5) - $cd0_{sub}$', fontsize=22)
    ax1.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax1.set_ylabel(r'$C_D$', fontsize=20)
    ax1.legend(fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    ax2 = plt.subplot(gs[0, 1])
    for kd_sub in kd_sub_range:
        C_Ds_sub = [rocket_CD(alpha, 0.5, kd_subsonic=kd_sub) for alpha in alphas]
        ax2.plot(np.rad2deg(alphas), C_Ds_sub, label=f'$kd_{{sub}}={kd_sub}$', linewidth=2)
    ax2.set_title('Subsonic (M = 0.5) - $kd_{sub}$', fontsize=22)
    ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax2.set_ylabel(r'$C_D$', fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # Supersonic plots
    ax3 = plt.subplot(gs[1, 0])
    for cd0_sup in cd0_sup_range:
        C_Ds_sup = [rocket_CD(alpha, 2.0, cd0_supersonic=cd0_sup) for alpha in alphas]
        ax3.plot(np.rad2deg(alphas), C_Ds_sup, label=f'$cd0_{{sup}}={cd0_sup}$', linewidth=2)
    ax3.set_title('Supersonic (M = 2.0) - $cd0_{sup}$', fontsize=22)
    ax3.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax3.set_ylabel(r'$C_D$', fontsize=20)
    ax3.legend(fontsize=20)
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=18)

    ax4 = plt.subplot(gs[1, 1])
    for kd_sup in kd_sup_range:
        C_Ds_sup = [rocket_CD(alpha, 2.0, kd_supersonic=kd_sup) for alpha in alphas]
        ax4.plot(np.rad2deg(alphas), C_Ds_sup, label=f'$kd_{{sup}}={kd_sup}$', linewidth=2)
    ax4.set_title('Supersonic (M = 2.0) - $kd_{sup}$', fontsize=22)
    ax4.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax4.set_ylabel(r'$C_D$', fontsize=20)
    ax4.legend(fontsize=20)
    ax4.grid(True)
    ax4.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/Sizing/CD_sensitivity_analysis.png')
    plt.close()

if __name__ == '__main__':
    machs = np.linspace(0.1, 2.0, 100)
    alphas = np.linspace(-math.radians(20), math.radians(20), 100)
    C_Ls = np.zeros((len(machs), len(alphas)))
    C_Ds = np.zeros((len(machs), len(alphas)))

    for i in range(len(machs)):
        for j in range(len(alphas)):
            C_Ls[i, j] = rocket_CL(alphas[j], machs[i])
            C_Ds[i, j] = rocket_CD(alphas[j], machs[i])

    # Interpolate to get values at specific points
    # Example: get values at M = 0.5, 0.8, 1.0, 1.2, 2.0
    M_values = [0.5, 0.8, 1.0, 1.2, 2.0]
    alpha_values = np.linspace(-math.radians(20), math.radians(20), 100)
    C_Ls_interp = RegularGridInterpolator((machs, alphas), C_Ls)
    C_Ds_interp = RegularGridInterpolator((machs, alphas), C_Ds)

    for M in M_values:
        C_Ls_M = C_Ls_interp((M, alpha_values))
        C_Ds_M = C_Ds_interp((M, alpha_values))
        
    kl_sub_range = np.linspace(1.5, 2.5, 5)
    kl_sup_range = np.linspace(0.8, 1.2, 5)
    plot_sensitivity_analysis(machs, alphas, kl_sub_range, kl_sup_range)

    plt.figure(figsize=(20, 10))
    plt.suptitle('Aerodynamic coefficients', fontsize=24)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    for M in M_values:
        C_Ls_M = C_Ls_interp((M, alpha_values))
        ax1.plot(np.rad2deg(alpha_values), C_Ls_M, label=f'{M}', linewidth=2)
    ax1.set_ylabel(r'$C_L$', fontsize=20)
    ax1.legend(fontsize=20)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2 = plt.subplot(gs[1])
    for M in M_values:
        C_Ds_M = C_Ds_interp((M, alpha_values))
        ax2.plot(np.rad2deg(alpha_values), C_Ds_M, label=f'{M}', linewidth=2)
    ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax2.set_ylabel(r'$C_D$', fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('results/Sizing/aerodynamic_coefficients_mach_sensitivity.png')
    plt.close()

    cd0_sub_range = np.linspace(0.04, 0.06, 5)
    kd_sub_range = np.linspace(0.4, 0.6, 5)
    cd0_sup_range = np.linspace(0.08, 0.12, 5)
    kd_sup_range = np.linspace(0.8, 1.2, 5)
    plot_drag_sensitivity_analysis(alphas, cd0_sub_range, kd_sub_range, cd0_sup_range, kd_sup_range)