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
from params import initial_conditions_dictionary, target_altitude_vertical_rising
print("\nInitial Conditions:")
print(initial_conditions_dictionary)

initial_state_vertical_rising = [
    initial_conditions_dictionary["initial_position_vector"][0],
    initial_conditions_dictionary["initial_position_vector"][1],
    initial_conditions_dictionary["initial_position_vector"][2],
    initial_conditions_dictionary["initial_velocity_vector"][0],
    initial_conditions_dictionary["initial_velocity_vector"][1],
    initial_conditions_dictionary["initial_velocity_vector"][2],
    initial_mass
]
print(f'\nInitial State for Vertical Rising: {initial_state_vertical_rising}')
from endo_atmosphere_vertical_rising import end_atmospheric_vertical_rising
vertical_rising_time, vertical_rising_states, \
      vertical_rising_final_state = end_atmospheric_vertical_rising(initial_state_vertical_rising,
                                    target_altitude_vertical_rising,
                                    stage_properties_dict['burn_out_masses'][0],
                                    stage_properties_dict['mass_flow_rates'][0],
                                    plot_bool = True)