import numpy as np
import matplotlib.pyplot as plt

def endo_atmospheric_model(alt):
    # Fundamental constants
    R = 287.05
    g0 = 9.80665
    gamma = 1.4
    
    # Each tuple: (lower_bound [m], upper_bound [m], 
    #             base_temp [K], base_pres [Pa], lapse_rate [K/m])
    # https://en.wikipedia.org/wiki/International_Standard_Atmosphere
    # Data derived from the standard atmosphere table
    # 0–11 km: Troposphere
    # 11–20 km: Tropopause (isothermal)
    # 20–32 km: Stratosphere (lapse rate = -1.0 °C/km)
    # 32–47 km: Stratosphere (lapse rate = -2.8 °C/km)
    # 47–51 km: Stratopause (isothermal)
    # 51–71 km: Mesosphere (lapse rate = +2.8 °C/km)
    # 71–86 km: Mesosphere (lapse rate = +2.0 °C/km)
    
    layers = [
        (    0.0, 11000.0, 288.15, 101325.0,   -0.0065),
        (11000.0, 20000.0, 216.65, 22632.0,     0.0  ),
        (20000.0, 32000.0, 216.65, 5474.9,   -0.0010),
        (32000.0, 47000.0, 228.65, 868.02,   -0.0028),
        (47000.0, 51000.0, 270.65, 110.91,     0.0  ),
        (51000.0, 71000.0, 270.65, 66.939,    0.0028),
        (71000.0, 86000.0, 214.65, 3.9564,    0.0020),
    ]
    
    # Clamp altitude if exceeding table range
    if alt < 0:
        alt = 0
    if alt > 86000.0:
        alt = 86000.0

    # Find the relevant layer
    for i in range(len(layers)):
        (h_base, h_top, T_base, P_base, alpha) = layers[i]
        
        # If altitude is in this layer
        if (alt >= h_base) and (alt <= h_top):
            if alpha != 0.0:
                # Temperature with lapse rate
                T = T_base + alpha * (alt - h_base)
                # Pressure with lapse rate
                P = P_base * (T / T_base) ** (-g0 / (alpha * R))
            else:
                # Isothermal layer
                T = T_base
                P = P_base * np.exp(-g0 * (alt - h_base) / (R * T))
            
            rho = P / (R * T)
            a = np.sqrt(gamma * R * T)
            return rho, P, a

def gravity_model_endo(altitude):
    R = 6371000                             # Earth radius [m]
    g0 = 9.80665                            # Gravity constant on Earth [m/s^2]
    g = g0 * (R / (R + altitude)) ** 2
    return g

if __name__ == '__main__':
    # Plot the atmospheric model
    save_path = 'results/figures/Atmosphere/'
    from tqdm import tqdm

    altitudes = np.linspace(0, 86000, 1000)

    rho_values = []
    p_values = []
    a_values = []
    g_values = []


    layers = [
        (    0.0, 11000.0, 288.15, 101325.0,   -0.0065),
        (11000.0, 20000.0, 216.65, 22632.0,     0.0  ),
        (20000.0, 32000.0, 216.65, 5474.9,   -0.0010),
        (32000.0, 47000.0, 228.65, 868.02,   -0.0028),
        (47000.0, 51000.0, 270.65, 110.91,     0.0  ),
        (51000.0, 71000.0, 270.65, 66.939,    0.0028),
        (71000.0, 86000.0, 214.65, 3.9564,    0.0020),
    ]

    for alt in tqdm(altitudes):
        rho, p, a = endo_atmospheric_model(alt)
        rho_values.append(rho)
        p_values.append(p)
        a_values.append(a)
        g_values.append(gravity_model_endo(alt))

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(altitudes, rho_values)
    axs[0].axvline(x=layers[0][1], color='r', linestyle='--', label='Troposphere')
    axs[0].axvline(x=layers[1][1], color='g', linestyle='--', label='Tropopause')   
    axs[0].axvline(x=layers[2][1], color='b', linestyle='--' , label='Stratosphere')
    axs[0].axvline(x=layers[3][1], color='y', linestyle='--', label='Stratopause')
    axs[0].axvline(x=layers[4][1], color='m', linestyle='--', label='Mesosphere')
    axs[0].axvline(x=layers[5][1], color='c', linestyle='--', label='Mesopause')
    axs[0].set_title('Density vs Altitude')
    axs[0].set_xlabel('Altitude [m]')
    axs[0].set_ylabel('Density [kg/m^3]')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(altitudes, p_values)
    axs[1].axvline(x=layers[0][1], color='r', linestyle='--')
    axs[1].axvline(x=layers[1][1], color='g', linestyle='--')   
    axs[1].axvline(x=layers[2][1], color='b', linestyle='--')
    axs[1].axvline(x=layers[3][1], color='y', linestyle='--')
    axs[1].axvline(x=layers[4][1], color='m', linestyle='--')
    axs[1].axvline(x=layers[5][1], color='c', linestyle='--')
    axs[1].set_title('Pressure vs Altitude')
    axs[1].set_xlabel('Altitude [m]')
    axs[1].set_ylabel('Pressure [Pa]')
    axs[1].grid()

    axs[2].plot(altitudes, a_values)
    axs[2].axvline(x=layers[0][1], color='r', linestyle='--')
    axs[2].axvline(x=layers[1][1], color='g', linestyle='--')   
    axs[2].axvline(x=layers[2][1], color='b', linestyle='--')
    axs[2].axvline(x=layers[3][1], color='y', linestyle='--')
    axs[2].axvline(x=layers[4][1], color='m', linestyle='--')
    axs[2].axvline(x=layers[5][1], color='c', linestyle='--')
    axs[2].set_title('Speed of Sound vs Altitude')
    axs[2].set_xlabel('Altitude [m]')
    axs[2].set_ylabel('Speed of Sound [m/s]')
    axs[2].grid()

    axs[3].plot(altitudes, g_values)
    axs[3].axvline(x=layers[0][1], color='r', linestyle='--')
    axs[3].axvline(x=layers[1][1], color='g', linestyle='--')   
    axs[3].axvline(x=layers[2][1], color='b', linestyle='--')
    axs[3].axvline(x=layers[3][1], color='y', linestyle='--')
    axs[3].axvline(x=layers[4][1], color='m', linestyle='--')
    axs[3].axvline(x=layers[5][1], color='c', linestyle='--')
    axs[3].set_title('Gravity vs Altitude')
    axs[3].set_xlabel('Altitude [m]')
    axs[3].set_ylabel('Gravity [m/s^2]')
    axs[3].grid()

    plt.tight_layout()
    plt.savefig(save_path + 'AtmosphereModel.png')
    plt.show()

