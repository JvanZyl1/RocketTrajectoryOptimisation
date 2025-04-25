import numpy as np
import matplotlib.pyplot as plt

def endo_atmospheric_model(alt, test_bool = False):
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
        (    0.0, 11000.0, 288.15, 101325.0,  0.0065),   # Troposphere
        (11000.0, 20000.0, 216.65, 22632.0,    0.0   ),   # Tropopause
        (20000.0, 32000.0, 216.65, 5474.9,    -0.0010),   # Stratosphere (lower)
        (32000.0, 47000.0, 228.65, 868.02,    -0.0028),   # Stratosphere (upper)
        (47000.0, 51000.0, 270.65, 110.91,     0.0   ),   # Stratopause
        (51000.0, 71000.0, 270.65, 66.939,     0.0028),   # Mesosphere (lower)
        (71000.0, 84852.0, 214.65, 3.9564,     0.0020),   # Mesosphere (upper)
    ]

    
    # Clamp altitude if exceeding table range
    if alt < 0:
        alt = 0
    if alt > 84852.0:
        print(f'Entering exo-atmosphere, change model.')
        rho = 0.0
        P = 0.0
        a = 0.0
        T = 186.946
        if test_bool:
                return rho, P, a, T
        else:
            return rho, P, a

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
            if test_bool:
                return rho, P, a, T
            else:
                return rho, P, a

def gravity_model_endo(altitude):
    R = 6371000                             # Earth radius [m]
    g0 = 9.80665                            # Gravity constant on Earth [m/s^2]
    g = g0 * (R / (R + altitude)) ** 2
    return g

def test_atmosphere_model():
    tol = 1e-2  # acceptable relative tolerance

    # Known values from standard atmosphere tables (ISA)
    # Format: (Altitude [m], Density [kg/m^3], Pressure [Pa], expected_a [m/s], expected_T [K])
    test_cases = [
        (0.0,     1.225,     101325.0, 340.3, 288.15),    # Sea level
        (11000.0, 0.36391,   22632.0,  295.1, 216.65),    # Tropopause
        (20000.0, 0.08803,   5474.9,   295.1, 216.65),    # Strat. lower
        (32000.0, 0.01322,   868.02,   301.6, 228.65),    # Strat. upper
        (47000.0, 0.00143,   110.91,   329.8, 270.65),    # Stratopause
        (71000.0, 0.000064,  3.9564,   295.0, 214.65),    # Mesopause
    ]

    for alt, rho_ref, p_ref, a_ref, T_ref in test_cases:
        rho, p, a, T = endo_atmospheric_model(alt, test_bool=True)

        assert abs(rho - rho_ref) / rho_ref < tol, f"rho mismatch at {alt} m, as {rho} instead of {rho_ref}"
        assert abs(p - p_ref) / p_ref < tol, f"pressure mismatch at {alt} m, as {p} instead of {p_ref}"
        assert abs(a - a_ref) / a_ref < 0.01, f"speed of sound mismatch at {alt} m, as {a} instead of {a_ref}"
        assert abs(T - T_ref) / T_ref < 0.005, f"temperature mismatch at {alt} m, as {T} instead of {T_ref}"

    print("All atmosphere model tests passed.")

if __name__ == '__main__':
    # Plot the atmospheric model
    save_path = 'results/ISAAtmosphere/'
    from tqdm import tqdm

    altitudes = np.linspace(0, 86000, 1000)

    rho_values = []
    p_values = []
    a_values = []
    g_values = []
    T_values = []

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
        rho, p, a, T = endo_atmospheric_model(alt, test_bool=True)
        rho_values.append(rho)
        p_values.append(p)
        a_values.append(a)
        g_values.append(gravity_model_endo(alt))
        T_values.append(T)

    T_values_degC = np.array(T_values) - 273.15
    # Now plot x as density and y as altitude
    plt.figure(figsize=(20, 10))
    plt.suptitle('International Standard Atmosphere', fontsize=22)
    plt.subplot(1,4,1)
    plt.plot(rho_values, altitudes/1000, linewidth=5)
    # Horizontal lines for each layer
    plt.axhline(y=layers[0][1]/1000, color='r', linestyle='--', label='Troposphere')
    plt.axhline(y=layers[1][1]/1000, color='g', linestyle='--', label='Tropopause')
    plt.axhline(y=layers[2][1]/1000, color='b', linestyle='--', label='Stratosphere')
    plt.axhline(y=layers[3][1]/1000, color='y', linestyle='--', label='Stratopause')
    plt.axhline(y=layers[4][1]/1000, color='m', linestyle='--', label='Mesosphere')
    plt.axhline(y=layers[5][1]/1000, color='c', linestyle='--', label='Mesopause')
    plt.xlabel('Density [kg/m^3]', fontsize=20)
    plt.ylabel('Altitude [km]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)

    plt.subplot(1,4,2)
    plt.plot(np.array(p_values)/1000, altitudes/1000, linewidth=5)
    plt.axhline(y=layers[0][1]/1000, color='r', linestyle='--', label='Troposphere')
    plt.axhline(y=layers[1][1]/1000, color='g', linestyle='--', label='Tropopause')
    plt.axhline(y=layers[2][1]/1000, color='b', linestyle='--', label='Stratosphere')
    plt.axhline(y=layers[3][1]/1000, color='y', linestyle='--', label='Stratopause')
    plt.axhline(y=layers[4][1]/1000, color='m', linestyle='--', label='Mesosphere')
    plt.axhline(y=layers[5][1]/1000, color='c', linestyle='--', label='Mesopause')
    plt.xlabel('Pressure [kPa]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()

    plt.subplot(1,4,3)
    plt.plot(a_values, altitudes/1000, linewidth=5)
    plt.axhline(y=layers[0][1]/1000, color='r', linestyle='--', label='Troposphere')
    plt.axhline(y=layers[1][1]/1000, color='g', linestyle='--', label='Tropopause')
    plt.axhline(y=layers[2][1]/1000, color='b', linestyle='--', label='Stratosphere')
    plt.axhline(y=layers[3][1]/1000, color='y', linestyle='--', label='Stratopause')
    plt.axhline(y=layers[4][1]/1000, color='m', linestyle='--', label='Mesosphere')
    plt.axhline(y=layers[5][1]/1000, color='c', linestyle='--', label='Mesopause')
    plt.xlabel('Speed of Sound [m/s]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()

    plt.subplot(1,4,4)
    plt.plot(T_values_degC, altitudes/1000, linewidth=5)
    plt.axhline(y=layers[0][1]/1000, color='r', linestyle='--', label='Troposphere')
    plt.axhline(y=layers[1][1]/1000, color='g', linestyle='--', label='Tropopause')
    plt.axhline(y=layers[2][1]/1000, color='b', linestyle='--', label='Stratosphere')
    plt.axhline(y=layers[3][1]/1000, color='y', linestyle='--', label='Stratopause')
    plt.axhline(y=layers[4][1]/1000, color='m', linestyle='--', label='Mesosphere')
    plt.axhline(y=layers[5][1]/1000, color='c', linestyle='--', label='Mesopause')
    plt.xlabel('Temperature [°C]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    
    plt.savefig(save_path + 'ISA.png')
    plt.close()

    test_atmosphere_model()

