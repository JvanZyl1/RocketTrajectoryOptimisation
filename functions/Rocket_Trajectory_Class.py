import numpy as np
import math
from functions.staging import staging_expendable, compute_stage_properties

from functions.params import write_params_to_json
write_params_to_json()

print_bool = False

# Call staging function
from functions.params import number_of_stages, specific_impulses_vacuum, structural_coefficients, payload_mass, semi_major_axis
delta_v_losses = 4200  # Delta_V for losses [m/s]
initial_mass, sub_stage_masses, stage_masses, structural_masses, \
      propellant_masses, delta_v_required, delta_v_required_stages, payload_ratios, mass_ratios = staging_expendable(number_of_stages,
                                                                                                           specific_impulses_vacuum,
                                                                                                          structural_coefficients,
                                                                                                          payload_mass,
                                                                                                          semi_major_axis,
                                                                                                          delta_v_losses)


class rocket_trajectory_optimiser:
    def __init__(self):