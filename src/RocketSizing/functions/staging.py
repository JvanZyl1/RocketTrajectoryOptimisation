import numpy as np
from math import sqrt, log, exp
from typing import Sequence, Dict, Tuple
from scipy.optimize import minimize, brentq, fsolve
import csv
# ────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────
MU_EARTH = 3.98602e14     # m3 s-2 – Earth gravitational parameter

# ────────────────────────────────────────────────────────────
#  FUNCTIONS
# ────────────────────────────────────────────────────────────

def staging_p1_reproduction(a: float,
                     m_pay: float,
                     dv_loss_a: Sequence[float],
                     dv_loss_d_1: float,
                     dv_d_1: float,
                     v_ex: Sequence[float],
                     eps: Sequence[float],
                     debug_bool: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    v_ex = np.asarray(v_ex, float)
    eps = np.asarray(eps, float)
    dv_loss_a = np.asarray(dv_loss_a, float)
    # 1 Calculate eps_d_1
    eps_d_1 = exp(-dv_d_1/v_ex[0])

    # 2 Calculate delta_v_req
    delta_v_req = sqrt(MU_EARTH/a)

    # ── 3  κ from Eq. 31 ───────────────────────────────────────────
    v_ex_1 = float(v_ex[0])
    eps_1 = float(eps[0])
    eps_a_1 = eps_1/eps_d_1
    v_ex_2 = float(v_ex[1])
    eps_2 = float(eps[1])
    dv_loss_a_1 = float(dv_loss_a[0])
    dv_loss_a_2 = float(dv_loss_a[1])
    def root(k: float) -> float:
        # Handle both array and scalar inputs
        if hasattr(k, "__len__"):
            k = float(k[0])
            
        # Make sure k is within bounds
        if k <= 0 or k >= min(v_ex_1, v_ex_2):
            return float('inf')
            
        # use *nominal* eps, not eps_a, in the logarithm (see Jo & Ahn, Eq. 31)
        try:
            # Add safety checks for the logarithm arguments
            term1 = (v_ex_1 - k)/(v_ex_1 * eps_1)
            term2 = (v_ex_2 - k)/(v_ex_2 * eps_2)
            if term1 <= 0 or term2 <= 0:
                return float('inf')
            lhs = v_ex_1 * log(term1) + v_ex_2 * log(term2)
            rhs = delta_v_req + dv_d_1
            ans = lhs - rhs
            return ans
        except (ValueError, ZeroDivisionError):
            return float('inf')

    # ---- kappa solution using fsolve ----------------------------------------    
    # Use a single initial guess in middle of valid range
    max_bound = min(v_ex_1, v_ex_2)
    initial_guess = 0.5 * max_bound
    
    # Define a simple function for fsolve
    def scalar_root(k):
        return root(k)
    
    # Use fsolve with relaxed tolerances
    result = fsolve(scalar_root, [initial_guess], 
                   xtol=1e-7,  # Relaxed tolerance
                   maxfev=1000)  # More iterations
    
    kappa = float(result[0])
    residual = abs(scalar_root(kappa))
    
    # 4 Size the expendable stage                                                                                                 
    lambda_2_star = kappa * eps_2 / ((1 - eps_2) * v_ex_2 - kappa)
    delta_v_2_star = v_ex_2 * log((1 + lambda_2_star)/(eps_2 + lambda_2_star))
    delta_v_2 = delta_v_2_star + dv_loss_a_2
    lambda_2_l_star = (eps_2 * exp(delta_v_2/v_ex_2) - 1) / (1 - exp(delta_v_2/v_ex_2))
    m_stage_2 = m_pay/lambda_2_l_star
    ms_2 = eps_2/lambda_2_l_star * m_pay
    mp_2 = (1 - eps_2)/lambda_2_l_star * m_pay
    eps_2_check = ms_2/(ms_2 + mp_2)
    assert abs(eps_2_check - eps_2) < 1e-6, f'eps_2_check: {eps_2_check} should equal eps_2: {eps_2}'

    # 5 Size the non-expendable stage
    eps_a_1 = eps_1/eps_d_1
    # correct (Eq. 22)
    lambda_1_star = kappa * eps_a_1/((1 - eps_a_1) * v_ex_1 - kappa)
    delta_v_a_1_star = v_ex_1 * log((1 + lambda_1_star)/(eps_a_1 + lambda_1_star))
    eps_d_1_l = exp(-(dv_d_1 + dv_loss_d_1)/v_ex_1)
    eps_a_1_l = eps_1/eps_d_1_l
    delta_v_a_1 = delta_v_a_1_star + dv_loss_a_1
    lambda_1_l_star = (eps_a_1_l * exp(delta_v_a_1/v_ex_1) - 1) / (1 - exp(delta_v_a_1/v_ex_1))
    m_stage_1 = (m_stage_2 + m_pay)/lambda_1_l_star
    ms_1 = eps_1/lambda_1_l_star * (m_stage_2 + m_pay)
    mp_1 = (1 - eps_1)/lambda_1_l_star * (m_stage_2 + m_pay)
    eps_1_check = ms_1/(ms_1 + mp_1)
    assert abs(eps_1_check - eps_1) < 1e-6, f'eps_1_check: {eps_1_check} should equal eps_1: {eps_1}'

    m0 = m_stage_1 + m_stage_2 + m_pay

    mp_d_1 = (1 - eps_d_1)/eps_d_1 * ms_1
    ms_d_1 = ms_1
    mp_a_1 = mp_1 - mp_d_1
    ms_a_1 = ms_1 + mp_d_1


    stage_masses_dict = {
        'structural_mass_stage_1_ascent': ms_a_1,
        'propellant_mass_stage_1_ascent': mp_a_1,
        'structural_mass_stage_1_descent': ms_d_1,
        'propellant_mass_stage_1_descent': mp_d_1,
        'structural_mass_stage_2_ascent': ms_2, # excludes payload mass
        'propellant_mass_stage_2_ascent': mp_2,
        'payload_mass': m_pay,
        'initial_mass': m0,
        'mass_at_stage_1_ascent_burnout': m0 - mp_a_1,
        'mass_of_rocket_at_stage_1_separation': m_stage_2 + m_pay,
        'mass_of_stage_1_at_separation': ms_d_1 + mp_d_1,
        'mass_at_stage_1_descent_burnout': ms_d_1,
        'mass_at_stage_2_ascent_burnout': ms_2 + m_pay,
        'mass_at_stage_2_separation': m_pay
    }

    trace = {
        "kappa": kappa,
        "payload_opt_1": lambda_1_star, "payload_opt_2": lambda_2_star,
        "payload_ratio_l_1_star": lambda_1_l_star, "payload_ratio_l_2_star": lambda_2_l_star,
        "eps_a_1": eps_a_1, "eps_d_1": eps_d_1,
        "eps_a_1_l": eps_a_1_l, "eps_d_1_l": eps_d_1_l,
        "dv_star_a_1": delta_v_a_1_star, "dv_star_2": delta_v_2_star,
        "m0": m0,
        "ms_1": ms_1, "mp_1": mp_1,
        "ms_2": ms_2, "mp_2": mp_2,
    }

    # Update loss-free velocity increments
    with open('data/rocket_parameters/velocity_increments.csv', 'a') as f:
        # clear the file
        f.truncate(0)
        writer = csv.writer(f)
        writer.writerow(['(sizing input) dv_loss_a_1', dv_loss_a_1])
        writer.writerow(['(sizing input) dv_loss_a_2', dv_loss_a_2])
        writer.writerow(['(sizing input) dv_loss_d_1', dv_loss_d_1])
        writer.writerow(['(sizing input) dv_d_1', dv_d_1])
        writer.writerow(['(sizing output) dv_star_a_1', delta_v_a_1_star])
        writer.writerow(['(sizing output) dv_star_2', delta_v_2_star])

    if debug_bool:
        return stage_masses_dict, trace
    else:
        return stage_masses_dict

# ────────────────────────────────────────────────────────────
#  REGRESSION CHECK – TABLE 1 ORIGINAL CONFIGURATION
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # inputs (Jo & Ahn 2021, Table 1)
    R_E, ALT = 6378137.0, 210000.0
    a        = R_E + ALT
    # Have to give initial guess
    '''
    eps_d_1 = 0.3443
    dv_d_1 = 3050.0 * log(1/eps_d_1)
    stage, tr = staging_p1_reproduction(
        a=a,
        m_pay = 15.5e3,
        dv_loss_a = [1396.0, 646.0],
        dv_loss_d_1 = 1475.0,
        dv_d_1 = dv_d_1,
        v_ex     =[3050.0, 3412.0],
        eps      =[0.05125, 0.0487]
    )
    '''
    eps_d_1 = 0.3606
    dv_d_1 = 3050.0 * log(1/eps_d_1)
    stage, tr = staging_p1_reproduction(
        a=a,
        m_pay = 15.5e3,
        dv_loss_a = [1391.0, 710.0],
        dv_loss_d_1 = 1401.0,
        dv_d_1 = dv_d_1,
        v_ex     =[3050.0, 3412.0],
        eps      =[0.05125, 0.0487],
        debug_bool = True
    )

    # reference targets (Table 1 and Table 2)
    ref = {
        "kappa":       2554.7,
        "dv_1_star_a":  1799.0,               # m/s (Table 1)
        "dv_2_star":    5600.0,               # m/s (Table 1)
        "dv_1_star_d":  1709.0,               # m/s (Table 1)
        "payload_ratio_1": 0.2323,      # Table 1
        "payload_ratio_2": 0.129,      # Table 1
        "payload_opt_1": 0.38623,
        "payload_opt_2": 0.1799, # NOT DONE, no losses
        "ms_1": 21.5e3,  # Table 1  
        "ms_2": 5.8e3,   # Table 1
        "mp_1": 398.6e3, # Table 1
        "mp_2": 114.2e3, # Table 1
        "eps_a_1":     0.1421, # Table 1
        "eps_d_1":     0.3606, # Table 1
        "m0":       555.6e3, # Table 1
    }

    print("\n─ INTERMEDIATE VALUES ─")
    # ALOT OF THESE CHECKS ARE WRONG
    def perecentage_difference(reference, calculated):
        return 100 * (calculated - reference) / reference
    # ── print comparison table ──────────────────────────────────────
    tolerance = 0.05                                   # 5 % for every quantity
    # kappa check
    print(f'Reference kappa: {ref["kappa"]} vs Calculated kappa {tr["kappa"]}, difference {perecentage_difference(ref["kappa"], tr["kappa"]):.2f}%')
    # Payload ratio check
    print(f'Reference payload ratio 1: {ref["payload_ratio_1"]} vs Calculated payload ratio 1 {tr["payload_opt_1"]:.2f}, difference {perecentage_difference(ref["payload_ratio_1"], tr["payload_opt_1"]):.2f}%')
    print(f'Reference payload ratio 2: {ref["payload_ratio_2"]} vs Calculated payload ratio 2 {tr["payload_opt_2"]:.2f}, difference {perecentage_difference(ref["payload_ratio_2"], tr["payload_opt_2"]):.2f}%')
    # Payload ratio with losses check
    print(f'Reference payload ratio 1 with losses: {ref["payload_ratio_1"]} vs Calculated payload ratio 1 with losses {tr["payload_ratio_l_1_star"]:.2f}, difference {perecentage_difference(ref["payload_ratio_1"], tr["payload_ratio_l_1_star"]):.2f}%')
    print(f'Reference payload ratio 2 with losses: {ref["payload_ratio_2"]} vs Calculated payload ratio 2 with losses {tr["payload_ratio_l_2_star"]:.2f}, difference {perecentage_difference(ref["payload_ratio_2"], tr["payload_ratio_l_2_star"]):.2f}%')
    # delta v star check
    print(f'Reference delta v star (ascent) 1: {ref["dv_1_star_a"]} vs Calculated delta v star (ascent) 1 {tr["dv_star_a_1"]}, difference {perecentage_difference(ref["dv_1_star_a"], tr["dv_star_a_1"]):.2f}%')
    print(f'Reference delta v star (ascent) 2: {ref["dv_2_star"]} vs Calculated delta v star (ascent) 2 {tr["dv_star_2"]}, difference {perecentage_difference(ref["dv_2_star"], tr["dv_star_2"]):.2f}%')
    # Structural masses
    print(f'Reference structural mass 1: {ref["ms_1"]} vs Calculated structural mass 1 {tr["ms_1"]}, difference {perecentage_difference(ref["ms_1"], tr["ms_1"]):.2f}%')
    print(f'Reference structural mass 2: {ref["ms_2"]} vs Calculated structural mass 2 {tr["ms_2"]}, difference {perecentage_difference(ref["ms_2"], tr["ms_2"]):.2f}%')
    # Propellant masses
    print(f'Reference propellant mass 1: {ref["mp_1"]} vs Calculated propellant mass 1 {tr["mp_1"]}, difference {perecentage_difference(ref["mp_1"], tr["mp_1"]):.2f}%')
    print(f'Reference propellant mass 2: {ref["mp_2"]} vs Calculated propellant mass 2 {tr["mp_2"]}, difference {perecentage_difference(ref["mp_2"], tr["mp_2"]):.2f}%')
    # Initial mass
    print(f'Reference initial mass: {ref["m0"]} vs Calculated initial mass {tr["m0"]}, difference {perecentage_difference(ref["m0"], tr["m0"]):.2f}%')
    # eps a
    print(f'Reference eps_a_1 {ref["eps_a_1"]} should equal {tr["eps_a_1"]} , difference {perecentage_difference(ref["eps_a_1"], tr["eps_a_1"]):.2f}%')