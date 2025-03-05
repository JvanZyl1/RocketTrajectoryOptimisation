import math

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
    
def compile_drag_coefficient_func(alpha):
    return lambda M: rocket_CD(alpha, M)