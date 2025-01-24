import numpy as np
from staging import staging

# Call staging function
from params import N, Isp, eps, m_pl, h_orbit
m0, m_subR, m_stg, m_str, m_prop, DV_req, DV_stg, lambda_vals, A = staging(N, Isp, eps, m_pl, h_orbit)

# Display outputs
print("Initial Mass [kg]:", m0)
print("Subrocket Masses [kg]:", m_subR)
print("Stages Masses [kg]:", m_stg)
print("Structural Masses [kg]:", m_str)
print("Propellant Masses [kg]:", m_prop)
print("Delta-V Required [m/s]:", DV_req)
print("Stage Delta-V [m/s]:", DV_stg)
print("Payload Ratios:", lambda_vals)
print("Mass Ratios:", A)
