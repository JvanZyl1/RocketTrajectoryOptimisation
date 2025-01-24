import numpy as np
from scipy.optimize import fsolve

from params import mu, R_earth, g0

def staging(N, Isp, eps, m_pl, h_orbit):
    """
    Provides the masses for the different stages.

    Parameters:
        N: int - Number of stages
        Isp: list - Specific impulse for each stage [s]
        eps: list - Structural coefficient for each stage
        m_pl: float - Payload mass [kg]
        h_orbit: float - Orbit altitude [m]
    
    Returns:
        Tuple containing:
        - Initial mass [kg]
        - Subrocket masses [kg]
        - Stage masses [kg]
        - Structural masses [kg]
        - Propellant masses [kg]
        - Delta-V required [m/s]
        - Stage Delta-V [m/s]
        - Payload ratio (lambda) for each stage
        - Mass ratio (A) for each stage
    """
    a = h_orbit + R_earth  # Semimajor axis [m]
    DV_loss = 2000  # Delta_V for losses [m/s]
    DV_req = np.sqrt(mu / a) + DV_loss  # Delta_V required [m/s]
    C = np.array(Isp) * g0  # Exhaust velocity [m/s]
    C_mean = np.mean(C)  # Mean characteristic velocity [m/s]
    eps_mean = np.mean(eps)  # Mean structural coefficient

    # Initial Lagrange Multiplier
    u = DV_req / C_mean
    lambda_tot = ((np.exp(-u / N) - eps_mean) / (1 - eps_mean))**N
    lambda_i = lambda_tot**(1 / N)
    Ai = 1 / (eps_mean * (1 - lambda_i) + lambda_i)
    p_init = 1 / (Ai * C_mean * eps_mean - C_mean)

    # Finding Lagrange Multiplier
    def func(p):
        result = -DV_req
        for i in range(N):
            result += C[i] * np.log((1 + p * C[i]) / (p * C[i] * eps[i]))
        return result

    p = fsolve(func, p_init)[0]

    A = np.zeros(N)
    lambda_vals = np.zeros(N)
    for i in range(N - 1, -1, -1):
        A[i] = (1 + p * C[i]) / (p * C[i] * eps[i])
        lambda_vals[i] = (1 - eps[i] * A[i]) / ((1 - eps[i]) * A[i])

    lambda_total = np.prod(lambda_vals)
    DV_stg = -C * np.log(eps * (1 - lambda_vals) + lambda_vals)

    # Error checking
    if abs(np.sum(DV_stg) - DV_req) > 1e-5:
        raise ValueError("Stage distribution computation error!")

    deriv2 = -(1 + p * C) / (A**2) + (eps / (1 - eps * A))**2
    if np.any(deriv2 < 0):
        raise ValueError("Check stages (there is max)")

    # Output masses
    m0 = m_pl / lambda_total  # Initial mass [kg]
    m_subR = np.zeros(N + 1)
    m_subR[-1] = m_pl

    m_stg = np.zeros(N)
    m_str = np.zeros(N)
    m_prop = np.zeros(N)

    for i in range(N - 1, -1, -1):
        m_subR[i] = m_subR[i + 1] / lambda_vals[i]  # Subrocket mass [kg]
        m_stg[i] = m_subR[i] - m_subR[i + 1]  # Stage mass [kg]
        m_str[i] = m_stg[i] * eps[i]  # Structural mass [kg]
        m_prop[i] = m_stg[i] - m_str[i]  # Propellant mass [kg]

    return m0, m_subR, m_stg, m_str, m_prop, DV_req, DV_stg, lambda_vals, A
