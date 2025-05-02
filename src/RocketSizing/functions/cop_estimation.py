import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def cop_func(L, alpha, M,
             baseline_x_cop = 0.8,
             x_cop_alpha_subsonic = 0.003,
             x_cop_machsupersonic = 0.1,
             x_cop_alpha_supersonic = 0.006):
    def cop_subsonic(L,
                 alpha,
                 baseline_x_cop=baseline_x_cop,
                 x_cop_alpha=x_cop_alpha_subsonic):
        x_fraction = baseline_x_cop + x_cop_alpha * alpha
        x_fraction = max(x_fraction, 0.2)
        x_fraction = min(x_fraction, 0.95)
        return L * x_fraction

    def cop_supersonic(L,
                    alpha,
                    M,
                    baseline_x_cop=baseline_x_cop,
                    x_cop_machsupersonic=x_cop_machsupersonic,
                    x_cop_alpha=x_cop_alpha_supersonic):
        # For M > 1.2, the CoP moves aft:
        x_fraction = baseline_x_cop - x_cop_machsupersonic * (M - 1.0) + x_cop_alpha * alpha
        x_fraction = max(x_fraction, 0.2)
        x_fraction = min(x_fraction, 0.95)
        return L * x_fraction


    def cop_transonic(L, alpha, M):
        # 0.8 - 1.2 Mach
        cop_sub = cop_subsonic(L, alpha, M)
        cop_super = cop_supersonic(L, alpha, M)
        cop_trans = cop_sub + (cop_super - cop_sub) * (M - 0.8) / 0.4
        return cop_trans

    if M < 0.8:
        return cop_subsonic(L, alpha)
    elif M < 1.2:
        return cop_transonic(L, alpha, M)
    else:
        return cop_supersonic(L, alpha, M)

def sensitivity_transonic():
    L = 70 #[m]
    machs = np.array([0.9, 1.0, 1.1, 1.2])
    alphas = np.linspace(-25, 25, 100)
    
    # Parameters to test sensitivity
    x_cop_alpha_subsonic = np.linspace(0.001, 0.005, 4)  # 4 values for 2x2 grid
    x_cop_machsupersonic = np.linspace(0.05, 0.15, 4)    # 4 values for 2x2 grid
    x_cop_alpha_supersonic = np.linspace(0.001, 0.01, 4) # 4 values for 2x2 grid
    
    # Setup figure for 2x2 grid
    plt.figure(figsize=(20, 15))
    plt.suptitle('Transonic CoP Sensitivity Analysis', fontsize=24)
    gs = gridspec.GridSpec(2, 2)
    
    # Analyze sensitivity to x_cop_alpha_subsonic at different mach numbers
    for i, mach in enumerate(machs[:4]):  # Ensure we don't exceed 4 values
        ax = plt.subplot(gs[i])
        for j, x_cop_sub in enumerate(x_cop_alpha_subsonic[:4]):  # Ensure we don't exceed 4 values
            cop_values = [cop_func(L, alpha, mach, x_cop_alpha_subsonic=x_cop_sub)/L for alpha in alphas]
            ax.plot(alphas, cop_values, label=fr'$\frac{{\partial x_{{cop}}}}{{\partial \alpha}}_{{sub}} = {x_cop_sub:.4f}$', linewidth=2)
        
        ax.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=16)
        ax.set_ylabel('CoP (length fraction)', fontsize=16)
        ax.set_title(f'Mach {mach}', fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/Sizing/CoP_transonic_sensitivity_analysis.png')
    plt.close()
    
    # Analyze sensitivity to x_cop_machsupersonic and x_cop_alpha_supersonic
    plt.figure(figsize=(20, 15))
    plt.suptitle('Transonic CoP Sensitivity to Supersonic Parameters', fontsize=24)
    gs = gridspec.GridSpec(2, 2)
    
    # Fixed alpha values for the plots
    fixed_alphas = [-10, 0, 10, 20]
    
    # First row: sensitivity to x_cop_machsupersonic
    for i in range(2):
        ax = plt.subplot(gs[i])
        alpha = fixed_alphas[i]
        for j, x_cop_mach in enumerate(x_cop_machsupersonic[:4]):
            cop_values = [cop_func(L, alpha, mach, x_cop_machsupersonic=x_cop_mach)/L for mach in np.linspace(0.8, 1.2, 100)]
            ax.plot(np.linspace(0.8, 1.2, 100), cop_values, 
                    label=fr'$\frac{{\partial x_{{cop}}}}{{\partial M}}_{{sup}} = {x_cop_mach:.2f}$', 
                    linewidth=2)
        
        ax.set_xlabel('Mach [-]', fontsize=16)
        ax.set_ylabel('CoP (length fraction)', fontsize=16)
        ax.set_title(fr'$\alpha = {alpha}^\circ$', fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Second row: sensitivity to x_cop_alpha_supersonic
    for i in range(2):
        ax = plt.subplot(gs[i+2])
        mach = machs[i+2]  # Use the last two mach numbers
        for j, x_cop_alpha_sup in enumerate(x_cop_alpha_supersonic[:4]):
            cop_values = [cop_func(L, alpha, mach, x_cop_alpha_supersonic=x_cop_alpha_sup)/L for alpha in alphas]
            ax.plot(alphas, cop_values, 
                    label=fr'$\frac{{\partial x_{{cop}}}}{{\partial \alpha}}_{{sup}} = {x_cop_alpha_sup:.4f}$', 
                    linewidth=2)
        
        ax.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=16)
        ax.set_ylabel('CoP (length fraction)', fontsize=16)
        ax.set_title(f'Mach {mach}', fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/Sizing/CoP_transonic_supersonic_params_sensitivity.png')
    plt.close()

def sensitivity_analysis():
    L = 70 #[m]
    M = 0.75 # independent variable
    alpha_s = np.linspace(-25, 25, 100)  # Ensure alpha_s is defined
    x_cop_alpha_subsonic = np.linspace(0.0001, 0.005, 6)  # Example range
    x_cop_alpha_supersonic = np.linspace(0.001, 0.01, 6)  # Example range
    x_cop_machsupersonic = np.linspace(0.05, 0.1, 6)  # Example range

    X, Y = np.meshgrid(alpha_s, x_cop_alpha_subsonic)
    cop_subsonic_sensitivity = np.zeros_like(X)
    for i in range(len(x_cop_alpha_subsonic)):
        for j in range(len(alpha_s)):
            cop_subsonic_sensitivity[i][j] = cop_func(L, alpha_s[j], M, x_cop_alpha_subsonic=x_cop_alpha_subsonic[i])

    plt.figure(figsize=(10, 6))
    for i in range(len(x_cop_alpha_subsonic)):
        plt.plot(alpha_s, cop_subsonic_sensitivity[i]/L, label=r'$\frac{\partial x_{cop}}{\partial \alpha}_{\text{sub}}$' + f'= {x_cop_alpha_subsonic[i]:.2e}', linewidth=2)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel('CoP (length fraction)', fontsize=20)
    plt.title(r'CoP variation with $\alpha$ at Mach = 0.75', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig('results/Sizing/CoP_subsonic_sensitivity_analysis.png')
    plt.close()

    # Sensitivity of supersonic flight to alpha
    M = 1.5
    cop_supersonic_sensitivity_alpha = np.zeros_like(X)
    for i in range(len(x_cop_alpha_supersonic)):
        for j in range(len(alpha_s)):
            cop_supersonic_sensitivity_alpha[i][j] = cop_func(L, alpha_s[j], M, x_cop_alpha_supersonic=x_cop_alpha_supersonic[i])

    # Sensitivity of supersonic flight to mach
    machs = np.linspace(1.2, 4.0, 10)
    alpha = 5  # Ensure alpha is a scalar for this part
    cop_supersonic_sensitivity_mach = np.zeros((len(x_cop_machsupersonic), len(machs)))
    for i in range(len(x_cop_machsupersonic)):
        for j in range(len(machs)):
            cop_supersonic_sensitivity_mach[i][j] = cop_func(L, alpha, machs[j], x_cop_machsupersonic=x_cop_machsupersonic[i])

    plt.figure(figsize=(20,15))
    plt.suptitle('CoP variation with Mach number and alpha', fontsize=24)
    gs = gridspec.GridSpec(2,1, height_ratios=[1,1])
    ax1 = plt.subplot(gs[0])
    for i in range(len(x_cop_machsupersonic)):
        ax1.plot(machs, cop_supersonic_sensitivity_mach[i]/L, label=r'$\frac{\partial x_{cop}}{\partial M}_{\text{sup}}$' + f'= {x_cop_machsupersonic[i]:.2e}', linewidth=2)
    ax1.set_xlabel('Mach [-]', fontsize=20)
    ax1.set_ylabel('CoP (length fraction)', fontsize=20)
    ax1.legend(fontsize=22)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_title(r'CoP variation with Mach number at $\alpha$ = 5', fontsize=24)
    ax2 = plt.subplot(gs[1])
    for i in range(len(x_cop_alpha_supersonic)):
        ax2.plot(alpha_s, cop_supersonic_sensitivity_alpha[i]/L, label=r'$\frac{\partial x_{cop}}{\partial \alpha}_{\text{sup}}$' + f'= {x_cop_alpha_supersonic[i]:.2e}', linewidth=2)
    ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax2.set_ylabel('CoP (length fraction)', fontsize=20)
    ax2.legend(fontsize=22)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.set_title(r'CoP variation with $\alpha$ at Mach = 1.5', fontsize=24)
    plt.savefig('results/Sizing/CoP_supersonic_sensitivity_analysis.png')
    plt.close()

    sensitivity_transonic()


def plot_cop_func():
    L = 70 #[m]

    # Define Mach and alpha ranges
    machs = [0.8, 1.0, 1.1, 1.2, 1.4, 1.6]
    alphas = np.linspace(-10, 10, 100)

    # Prepare figure
    plt.figure(figsize=(20, 15))
    plt.suptitle('CoP variation with Mach number and alpha', fontsize=24)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Plot CoP variation with alpha for different Mach numbers
    ax1 = plt.subplot(gs[0])
    for mach in machs:
        cop_norm = [cop_func(L, alpha, mach)/L for alpha in alphas]
        ax1.plot(alphas, cop_norm, label=f'Mach {mach}', linewidth=2)
    ax1.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=20)
    ax1.set_ylabel('CoP (length fraction)', fontsize=20)
    ax1.legend(fontsize=22)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_title('CoP variation with Mach number', fontsize=24)

    # Plot CoP variation with Mach for different alpha values
    alpha_values = [-10, -5, 0, 5, 10]
    mach_range = np.linspace(0.6, 2.0, 100)
    ax2 = plt.subplot(gs[1])
    for alpha in alpha_values:
        cop_norm = [cop_func(L, alpha, mach)/L for mach in mach_range]
        ax2.plot(mach_range, cop_norm, label=fr'$\alpha = {alpha}$', linewidth=2)
    ax2.set_xlabel('Mach [-]', fontsize=20)
    ax2.set_ylabel('CoP (length fraction)', fontsize=20)
    ax2.legend(fontsize=22)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.set_title('CoP variation with Mach number', fontsize=24)

    plt.savefig('results/Sizing/CoP_sensitivity_analysis.png')
    plt.close()

    sensitivity_analysis()

if __name__ == '__main__':
    plot_cop_func()