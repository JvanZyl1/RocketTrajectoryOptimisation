import numpy as np
import matplotlib.pyplot as plt

def cop_func(L, alpha, M):
    def cop_subsonic(L,
                 alpha,
                 baseline_x_cop=0.8,
                 x_cop_alpha=0.005):
        x_fraction = baseline_x_cop + x_cop_alpha * alpha
        return L * x_fraction

    def cop_supersonic(L,
                    alpha,
                    M,
                    baseline_x_cop=0.8,
                    x_cop_machsupersonic=0.1,
                    x_cop_alpha=0.002):
        # For M > 1.2, the CoP moves aft:
        x_fraction = baseline_x_cop - x_cop_machsupersonic * (M - 1.0) + x_cop_alpha * alpha
        x_fraction = max(x_fraction, 0.5)
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
    

def plot_cop_func():
    L = 70 #[m]

    # 3D plot mach vs alpha vs cop
    mach = np.linspace(0.6, 2.0, 100)
    alpha = np.linspace(-10, 10, 100)
    mach, alpha = np.meshgrid(mach, alpha)
    cop = np.zeros_like(mach)

    for i in range(len(mach)):
        for j in range(len(mach[i])):
            cop[i][j] = cop_func(L, alpha[i][j], mach[i][j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mach, alpha, cop, cmap='viridis')
    ax.set_xlabel('Mach')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('CoP')
    plt.close()

    # Now do 5 cuts of mach
    mach_new = [0.8, 1.0, 1.2, 1.4, 1.6]
    alpha = np.linspace(-10, 10, 100)
    cop_norm_08 = np.zeros_like(alpha)
    cop_norm_1 = np.zeros_like(alpha)
    cop_norm_12 = np.zeros_like(alpha)
    cop_norm_14 = np.zeros_like(alpha)
    cop_norm_16 = np.zeros_like(alpha)

    for i in range(len(alpha)):
        cop_norm_08[i] = cop_func(L, alpha[i], 0.8)/L
        cop_norm_1[i] = cop_func(L, alpha[i], 1.0)/L
        cop_norm_12[i] = cop_func(L, alpha[i], 1.2)/L
        cop_norm_14[i] = cop_func(L, alpha[i], 1.4)/L
        cop_norm_16[i] = cop_func(L, alpha[i], 1.6)/L

    save_path = 'results/Sizing/CoP_variation_constant_mach.png'
    fig, ax = plt.subplots()
    ax.plot(alpha, cop_norm_08, label='Mach 0.8')
    ax.plot(alpha, cop_norm_1, label='Mach 1.0')
    ax.plot(alpha, cop_norm_12, label='Mach 1.2')
    ax.plot(alpha, cop_norm_14, label='Mach 1.4')
    ax.plot(alpha, cop_norm_16, label='Mach 1.6')
    ax.set_xlabel('Alpha (degrees)')
    ax.set_ylabel('CoP (length fraction)')
    ax.legend()
    plt.savefig(save_path)
    plt.close()

    alpha_new = [-10, -5, 0, 5, 10]
    mach = np.linspace(0.6, 2.0, 100)
    cop_norm_neg10 = np.zeros_like(mach)
    cop_norm_neg5 = np.zeros_like(mach)
    cop_norm_0 = np.zeros_like(mach)
    cop_norm_5 = np.zeros_like(mach)
    cop_norm_10 = np.zeros_like(mach)

    for i in range(len(mach)):
        cop_norm_neg10[i] = cop_func(L, -10, mach[i])/L
        cop_norm_neg5[i] = cop_func(L, -5, mach[i])/L
        cop_norm_0[i] = cop_func(L, 0, mach[i])/L
        cop_norm_5[i] = cop_func(L, 5, mach[i])/L
        cop_norm_10[i] = cop_func(L, 10, mach[i])/L

    save_path = 'results/Sizing/CoP_variation_constant_alpha.png'
    fig, ax = plt.subplots()
    ax.plot(mach, cop_norm_neg10, label='Alpha -10')
    ax.plot(mach, cop_norm_neg5, label='Alpha -5')
    ax.plot(mach, cop_norm_0, label='Alpha 0')
    ax.plot(mach, cop_norm_5, label='Alpha 5')
    ax.plot(mach, cop_norm_10, label='Alpha 10')
    ax.set_xlabel('Mach')
    ax.set_ylabel('CoP (length fraction)')
    ax.legend()
    plt.savefig(save_path)
    plt.close()