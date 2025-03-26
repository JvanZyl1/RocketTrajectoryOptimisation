import math
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    mach_numbers = np.linspace(0.1, 2.0, 100)
    alphas = np.linspace(-math.radians(20), math.radians(20), 100)
    X, Y = np.meshgrid(mach_numbers, alphas)
    C_Ls = np.zeros_like(X)
    C_Ds = np.zeros_like(X)

    for i in range(len(alphas)):
        for j in range(len(mach_numbers)):
            C_Ls[i, j] = rocket_CL(alphas[i], mach_numbers[j])
            C_Ds[i, j] = rocket_CD(alphas[i], mach_numbers[j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, C_Ls)
    ax.set_xlabel('Mach number')
    ax.set_ylabel('Angle of attack (rad)')
    ax.set_zlabel('Lift coefficient')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, C_Ds)
    ax.set_xlabel('Mach number')
    ax.set_ylabel('Angle of attack (rad)')
    ax.set_zlabel('Drag coefficient')
    plt.show()

    C_Ls_05 = np.zeros_like(alphas)
    C_Ds_05 = np.zeros_like(alphas)

    for i in range(len(alphas)):
        C_Ls_05[i] = rocket_CL(alphas[i], 0.5)
        C_Ds_05[i] = rocket_CD(alphas[i], 0.5)

    C_Ls_08 = np.zeros_like(alphas)
    C_Ds_08 = np.zeros_like(alphas)

    for i in range(len(alphas)):
        C_Ls_08[i] = rocket_CL(alphas[i], 0.8)
        C_Ds_08[i] = rocket_CD(alphas[i], 0.8)

    C_Ls_1 = np.zeros_like(alphas)
    C_Ds_1 = np.zeros_like(alphas)

    for i in range(len(alphas)):
        C_Ls_1[i] = rocket_CL(alphas[i], 1)
        C_Ds_1[i] = rocket_CD(alphas[i], 1)

    C_Ls_12 = np.zeros_like(alphas)
    C_Ds_12 = np.zeros_like(alphas)

    for i in range(len(alphas)):
        C_Ls_12[i] = rocket_CL(alphas[i], 1.2)
        C_Ds_12[i] = rocket_CD(alphas[i], 1.2)

    C_Ls_2 = np.zeros_like(alphas)
    C_Ds_2 = np.zeros_like(alphas)

    for i in range(len(alphas)):
        C_Ls_2[i] = rocket_CL(alphas[i], 2)
        C_Ds_2[i] = rocket_CD(alphas[i], 2)

    save_path = 'results/Sizing/Cl_Cd_variation.png'

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.rad2deg(alphas), C_Ls_05, label='M = 0.5')
    plt.plot(np.rad2deg(alphas), C_Ls_08, label='M = 0.8')
    plt.plot(np.rad2deg(alphas), C_Ls_1, label='M = 1.0')
    plt.plot(np.rad2deg(alphas), C_Ls_12, label='M = 1.2')
    plt.plot(np.rad2deg(alphas), C_Ls_2, label='M = 2.0')
    plt.xlabel('Angle of attack (degrees)')
    plt.ylabel('Lift coefficient')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.rad2deg(alphas), C_Ds_05, label='M = 0.5')
    plt.plot(np.rad2deg(alphas), C_Ds_08, label='M = 0.8')
    plt.plot(np.rad2deg(alphas), C_Ds_1, label='M = 1.0')
    plt.plot(np.rad2deg(alphas), C_Ds_12, label='M = 1.2')
    plt.plot(np.rad2deg(alphas), C_Ds_2, label='M = 2.0')
    plt.xlabel('Angle of attack (degrees)')
    plt.ylabel('Drag coefficient')
    plt.legend()
    plt.savefig(save_path)
    plt.show()