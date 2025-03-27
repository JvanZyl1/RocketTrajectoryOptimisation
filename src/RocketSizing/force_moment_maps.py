import matplotlib.pyplot as plt
import numpy as np

# Rough constants
T_e = 2745*1000                                     # Engine thrust [N]
n_eg_g = 3 + 12                                     # Number of gimballed engines
n_eg_ng = 26                                        # Number of non-gimballed engines
p_e = 100000                                        # Nozzle exit pressure [Pa]
p_a = 101325                                        # Ambient pressure [Pa]
A_e = 0.01                                          # Engine nozzle exit area [m^2]

T_e_with_losses = T_e + (p_e - p_a) * A_e           # Engine thrust with losses [N]
T_g_full_throttle = T_e_with_losses * n_eg_g        # Gimballed thrust (full throttle) [N]
T_ng_full_throttle = T_e_with_losses * n_eg_ng      # Non-gimballed thrust (full throttle) [N]
I_z = 2e10                                          # Rocket inertia [kg m^2]
d_tcg = 60                                          # Distance from CG to thrust vector [m]


theta_gimbal = np.linspace(0, np.radians(30), 100)


# First change all engines with same throttle (0.7 -> 1) and gimbal angle (-30 -> 30) put 3D plots of how thrust // and _|_ and Mz change

def thrusts_and_moments(theta_gimbal, throttle_g, throttle_ng):
    T_g = T_g_full_throttle * throttle_g
    T_ng = T_ng_full_throttle * throttle_ng

    T_parallel = T_g * np.cos(theta_gimbal) + T_ng
    T_perpendicular = T_g * np.sin(theta_gimbal)
    M_z = d_tcg * T_perpendicular

    return T_parallel, T_perpendicular, M_z

def plot_thrusts_and_moments(theta_gimbals, T_parrallels, T_perpendiculars, M_zs, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15)) 
    axs[0].plot(np.degrees(theta_gimbals), T_parrallels)
    axs[0].set_xlabel('Gimbal Angle [deg]')
    axs[0].set_ylabel('Parallel Thrust [N]')
    axs[0].grid()
    axs[0].set_title(f'{title}')

    axs[1].plot(np.degrees(theta_gimbals), T_perpendiculars)
    axs[1].set_xlabel('Gimbal Angle [deg]')
    axs[1].set_ylabel('Perpendicular Thrust [N]')
    axs[1].grid()

    axs[2].plot(np.degrees(theta_gimbals), M_zs)
    axs[2].set_xlabel('Gimbal Angle [deg]')
    axs[2].set_ylabel('Moment [Nm]')
    axs[2].grid()

    plt.tight_layout()
    plt.show()

# Throttle = 1, gimbal = -30 -> 30
T_parrallels_test_1, T_perpendiculars_test_1, M_zs_test_1 = zip(*[thrusts_and_moments(angle, 1, 1) for angle in theta_gimbal])

# Throttle_ng = 1, Throttle_g = 0.7, gimbal = -30 -> 30
T_parrallels_test_2, T_perpendiculars_test_2, M_zs_test_2 = zip(*[thrusts_and_moments(angle, 0.7, 1) for angle in theta_gimbal])

# Throttle_ng = 1, Throttle_g = 0.8, gimbal = -30 -> 30
T_parrallels_test_3, T_perpendiculars_test_3, M_zs_test_3 = zip(*[thrusts_and_moments(angle, 0.8, 1) for angle in theta_gimbal])

# Throttle_ng = 1, Throttle_g = 0.9, gimbal = -30 -> 30
T_parrallels_test_4, T_perpendiculars_test_4, M_zs_test_4 = zip(*[thrusts_and_moments(angle, 0.9, 1) for angle in theta_gimbal])

# Throttle_ng = 0.7, Throttle_g = 1, gimbal = -30 -> 30
T_parrallels_test_5, T_perpendiculars_test_5, M_zs_test_5 = zip(*[thrusts_and_moments(angle, 1, 0.7) for angle in theta_gimbal])

# Throttle_ng = 0.8, Throttle_g = 1, gimbal = -30 -> 30
T_parrallels_test_6, T_perpendiculars_test_6, M_zs_test_6 = zip(*[thrusts_and_moments(angle, 1, 0.8) for angle in theta_gimbal])

# Throttle_ng = 0.9, Throttle_g = 1, gimbal = -30 -> 30
T_parrallels_test_7, T_perpendiculars_test_7, M_zs_test_7 = zip(*[thrusts_and_moments(angle, 1, 0.9) for angle in theta_gimbal])

# Throttle_ng = 0.7, Throttle_g = 0.7, gimbal = -30 -> 30
T_parrallels_test_8, T_perpendiculars_test_8, M_zs_test_8 = zip(*[thrusts_and_moments(angle, 0.7, 0.7) for angle in theta_gimbal])

# Throttle_ng = 0.7, Throttle_g = 0.8, gimbal = -30 -> 30
T_parrallels_test_9, T_perpendiculars_test_9, M_zs_test_9 = zip(*[thrusts_and_moments(angle, 0.8, 0.7) for angle in theta_gimbal])

# Throttle_ng = 0.7, Throttle_g = 0.9, gimbal = -30 -> 30
T_parrallels_test_10, T_perpendiculars_test_10, M_zs_test_10 = zip(*[thrusts_and_moments(angle, 0.9, 0.7) for angle in theta_gimbal])


# Throttle_ng = 0.8, Throttle_g = 0.8, gimbal = -30 -> 30
T_parrallels_test_11, T_perpendiculars_test_11, M_zs_test_11 = zip(*[thrusts_and_moments(angle, 0.8, 0.8) for angle in theta_gimbal])


# Throttle_ng = 0.8, Throttle_g = 0.9, gimbal = -30 -> 30
T_parrallels_test_12, T_perpendiculars_test_12, M_zs_test_12 = zip(*[thrusts_and_moments(angle, 0.9, 0.8) for angle in theta_gimbal])


# Throttle_ng = 0.9, Throttle_g = 0.9, gimbal = -30 -> 30
T_parrallels_test_13, T_perpendiculars_test_13, M_zs_test_13 = zip(*[thrusts_and_moments(angle, 0.9, 0.9) for angle in theta_gimbal])

fig, axs = plt.subplots(3, 1, figsize=(10, 15), squeeze=False)
axs = axs.flatten()  # This flattens the 2D array to a 1D array

axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_1, 'r', label='Throttle_ng = 1, Throttle_g = 1')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_2, 'g', label='Throttle_ng = 1, Throttle_g = 0.7')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_3, 'b', label='Throttle_ng = 1, Throttle_g = 0.8')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_4, 'y', label='Throttle_ng = 1, Throttle_g = 0.9')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_5, 'c', label='Throttle_ng = 0.7, Throttle_g = 1')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_6, 'm', label='Throttle_ng = 0.8, Throttle_g = 1')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_7, 'k', label='Throttle_ng = 0.9, Throttle_g = 1')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_8, 'r--', label='Throttle_ng = 0.7, Throttle_g = 0.7')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_9, 'g--', label='Throttle_ng = 0.7, Throttle_g = 0.8')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_10, 'b--', label='Throttle_ng = 0.7, Throttle_g = 0.9')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_11, 'y--', label='Throttle_ng = 0.8, Throttle_g = 0.8')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_12, 'c--', label='Throttle_ng = 0.8, Throttle_g = 0.9')
axs[0].plot(np.degrees(theta_gimbal), T_parrallels_test_13, 'm--', label='Throttle_ng = 0.9, Throttle_g = 0.9')
axs[0].set_xlabel('Gimbal Angle [deg]')
axs[0].set_ylabel('Parallel Thrust [N]')
axs[0].grid()
axs[0].set_title('Parallel Thrusts')
axs[0].legend()

axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_1, 'r', label='Throttle_ng = 1, Throttle_g = 1')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_2, 'g', label='Throttle_ng = 1, Throttle_g = 0.7')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_3, 'b', label='Throttle_ng = 1, Throttle_g = 0.8')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_4, 'y', label='Throttle_ng = 1, Throttle_g = 0.9')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_5, 'c', label='Throttle_ng = 0.7, Throttle_g = 1')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_6, 'm', label='Throttle_ng = 0.8, Throttle_g = 1')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_7, 'k', label='Throttle_ng = 0.9, Throttle_g = 1')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_8, 'r--', label='Throttle_ng = 0.7, Throttle_g = 0.7')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_9, 'g--', label='Throttle_ng = 0.7, Throttle_g = 0.8')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_10, 'b--', label='Throttle_ng = 0.7, Throttle_g = 0.9')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_11, 'y--', label='Throttle_ng = 0.8, Throttle_g = 0.8')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_12, 'c--', label='Throttle_ng = 0.8, Throttle_g = 0.9')
axs[1].plot(np.degrees(theta_gimbal), T_perpendiculars_test_13, 'm--', label='Throttle_ng = 0.9, Throttle_g = 0.9')

axs[1].set_xlabel('Gimbal Angle [deg]')
axs[1].set_ylabel('Perpendicular Thrust [N]')
axs[1].grid()
axs[1].set_title('Perpendicular Thrusts')
axs[1].legend()

axs[2].plot(np.degrees(theta_gimbal), M_zs_test_1, 'r', label='Throttle_ng = 1, Throttle_g = 1')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_2, 'g', label='Throttle_ng = 1, Throttle_g = 0.7')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_3, 'b', label='Throttle_ng = 1, Throttle_g = 0.8')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_4, 'y', label='Throttle_ng = 1, Throttle_g = 0.9')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_5, 'c', label='Throttle_ng = 0.7, Throttle_g = 1')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_6, 'm', label='Throttle_ng = 0.8, Throttle_g = 1')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_7, 'k', label='Throttle_ng = 0.9, Throttle_g = 1')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_8, 'r--', label='Throttle_ng = 0.7, Throttle_g = 0.7')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_9, 'g--', label='Throttle_ng = 0.7, Throttle_g = 0.8')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_10, 'b--', label='Throttle_ng = 0.7, Throttle_g = 0.9')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_11, 'y--', label='Throttle_ng = 0.8, Throttle_g = 0.8')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_12, 'c--', label='Throttle_ng = 0.8, Throttle_g = 0.9')
axs[2].plot(np.degrees(theta_gimbal), M_zs_test_13, 'm--', label='Throttle_ng = 0.9, Throttle_g = 0.9')

axs[2].set_xlabel('Gimbal Angle [deg]')
axs[2].set_ylabel('Moment [Nm]')
axs[2].grid()
axs[2].set_title('Moments')
axs[2].legend()

plt.tight_layout()
plt.show()