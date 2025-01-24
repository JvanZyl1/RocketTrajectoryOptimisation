import numpy as np
from staging import staging_expendable, compute_stage_properties

# Call staging function
from params import number_of_stages, specific_impulses_vacuum, structural_coefficients, payload_mass, semi_major_axis
delta_v_losses = 2000  # Delta_V for losses [m/s]
initial_mass, sub_stage_masses, stage_masses, structural_masses, \
      propellant_masses, delta_v_required, delta_v_required_stages, payload_ratios, mass_ratios = staging_expendable(number_of_stages,
                                                                                                          specific_impulses_vacuum,
                                                                                                          structural_coefficients,
                                                                                                          payload_mass,
                                                                                                          semi_major_axis,
                                                                                                          delta_v_losses)

# Display outputs
print("Initial Mass [kg]:", initial_mass)
print("Subrocket Masses [kg]:", sub_stage_masses) # Final is payload
print("Stages Masses [kg]:", stage_masses)
print("Structural Masses [kg]:", structural_masses)
print("Propellant Masses [kg]:", propellant_masses)
print("Delta-V Required [m/s]:", delta_v_required)
print("Stage Delta-V [m/s]:", delta_v_required_stages)
print("Payload Ratios:", payload_ratios)
print("Mass Ratios:", mass_ratios)

from params import maximum_accelerations, g0, mass_fairing

stage_properties_dict = compute_stage_properties(initial_mass,
                                                 stage_masses,
                                                 propellant_masses,
                                                 structural_masses,
                                                 specific_impulses_vacuum,
                                                 g0,
                                                 maximum_accelerations,
                                                 mass_fairing)

# Display outputs
print("\nStage Properties:")
print(stage_properties_dict)

# Load and display initial conditions
from params import initial_conditions_dictionary
print("\nInitial Conditions:")
print(initial_conditions_dictionary)

