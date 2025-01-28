import numpy as np

state_gravity_turn_final = [6411897,
                            158188.9,
                            409263,
                            412.4,
                            1968,
                            -3611,
                            2465] # [r, v, m]

# Orientation Vector Components
from functions.params import kick_angle
orientation_x = np.sin(kick_angle)
orientation_y = np.cos(kick_angle)
orientation_z = 0

# Orientation Vector
orientation_vector = np.array([orientation_x, orientation_y, orientation_z])
