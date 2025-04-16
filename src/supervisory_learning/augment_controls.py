import jax.numpy as jnp
import math
import numpy as np
def augment_targets_ascent(targets):
    # targets are [Mz, F_parallel, F_perpendicular]
    GimbalAngledeg = targets[:, 0]
    MVMThrottle = targets[:, 1]

    max_gimbal_angle_rad = math.radians(5)

    u0 = np.radians(GimbalAngledeg) / max_gimbal_angle_rad
    nominal_throttle = 0.5
    u1 = (MVMThrottle - nominal_throttle) / (1 - nominal_throttle) * 2 - 1

    u_array = jnp.array([u0, u1])
    return u_array.T


def deaugment_targets_ascent(targets):
    u0 = targets[:, 0]
    u1 = targets[:, 1]

    max_gimbal_angle_rad = math.radians(5)
    nominal_throttle = 0.5

    gimbal_angle_deg = np.degrees(u0 * max_gimbal_angle_rad)
    non_nominal_throttle = (u1 + 1) / 2
    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    targets_array = jnp.array([gimbal_angle_deg, throttle])
    return targets_array.T