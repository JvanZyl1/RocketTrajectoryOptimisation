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
from endo_atmosphere_vertical_rising import endo_atmospheric_vertical_rising
vertical_rising_time, vertical_rising_states, \
      vertical_rising_final_state = endo_atmospheric_vertical_rising(initial_state_vertical_rising,
                                    target_altitude_vertical_rising,
                                    stage_properties_dict['burn_out_masses'][0],
                                    stage_properties_dict['mass_flow_rates'][0],
                                    plot_bool = False)

from params import w_earth, kick_angle, unit_east_vector
print(f'vertical_rising_final_state: {vertical_rising_final_state}')

vertical_rising_final_position_vector = vertical_rising_final_state[:3]
vertical_rising_final_velocity_vector = vertical_rising_final_state[3:6]
vertical_rising_final_mass = vertical_rising_final_state[6]
vertical_rising_final_time = vertical_rising_time[-1]


vertical_rising_final_relative_velocity_vector = vertical_rising_final_velocity_vector - np.cross(w_earth, vertical_rising_final_position_vector)
vertical_rising_final_relative_velocity_magnitude = np.linalg.norm(vertical_rising_final_relative_velocity_vector)
gravity_turn_relative_velocity_magnitude = vertical_rising_final_relative_velocity_magnitude * np.sin(kick_angle)
eastward_velocity = gravity_turn_relative_velocity_magnitude * unit_east_vector  # Eastward velocity component [m/s]
initial_relative_velocity_GT = vertical_rising_final_relative_velocity_vector + eastward_velocity  # Initial relative velocity in ground tracking frame [m/s]
initial_velocity_GT = initial_relative_velocity_GT + np.cross(w_earth, vertical_rising_final_position_vector)  # Initial velocity in ground tracking frame [m/s]

# Print all of above
print(f'vertical_rising_final_position_vector: {vertical_rising_final_position_vector}'
        f'\n vertical_rising_final_velocity_vector: {vertical_rising_final_velocity_vector}'
        f'\n vertical_rising_final_mass: {vertical_rising_final_mass}'
        f'\n vertical_rising_final_time: {vertical_rising_final_time}'
        f'\n vertical_rising_final_relative_velocity_vector: {vertical_rising_final_relative_velocity_vector}'
        f'\n vertical_rising_final_relative_velocity_magnitude: {vertical_rising_final_relative_velocity_magnitude}'
        f'\n gravity_turn_relative_velocity_magnitude: {gravity_turn_relative_velocity_magnitude}'
        f'\n eastward_velocity: {eastward_velocity}'
        f'\n initial_relative_velocity_GT: {initial_relative_velocity_GT}'
        f'\n initial_velocity_GT: {initial_velocity_GT}')


initial_state_GT = np.concatenate((vertical_rising_final_position_vector,
                                   initial_velocity_GT,
                                   [vertical_rising_final_mass]))  # Initial state for ground tracking

initial_state_GT = np.array([6351985,
                             4058.29,
                             578075.959,
                             23.164,
                             463.236,
                             2.10817,
                             25208])

print(f'initial_state_GT: {initial_state_GT}')

# Implement gravity turn with a gradual kick-angle change.
from endo_atmosphere_gravity_turn import endo_atmospheric_gravity_turn
from params import target_altitude_gravity_turn, R_earth

gravity_turn_time, gravity_turn_states, \
      gravity_turn_final_state = endo_atmospheric_gravity_turn(t_start = vertical_rising_final_time,
                                    initial_state = initial_state_GT,
                                    target_altitude = target_altitude_gravity_turn,
                                    minimum_mass=stage_properties_dict['burn_out_masses'][0],
                                    mass_flow_endo=stage_properties_dict['mass_flow_rates'][0],
                                    plot_bool = False)
gravity_turn_final_position_vector = gravity_turn_final_state[:3]
gravity_turn_final_velocity_vector = vertical_rising_final_state[3:6]
gravity_turn_final_mass = gravity_turn_final_state[6]
gravity_turn_final_time = gravity_turn_time[-1]

gravity_turn_final_altitude = np.linalg.norm(gravity_turn_final_position_vector) - R_earth

print(f'gravity_turn_final_state: {gravity_turn_final_state}'
      f'\n altitude: {gravity_turn_final_altitude}'
        f'\n target altitude: {target_altitude_gravity_turn}'
        f'\n minimum mass: {stage_properties_dict["burn_out_masses"][0]}')

### Coasting procedure
# The rocket first stage has departed.
# The fairing along with it.
mass_coasting = sub_stage_masses[1] - mass_fairing # Mass of the second stage & payload - fairing [kg]

from endo_atmosphere_coasting import endo_atmosphere_coasting
from params import coasting_time

coasting_time, coasting_states, \
        coasting_final_state = endo_atmosphere_coasting(t_start = gravity_turn_final_time,
                                        initial_state = gravity_turn_final_state,
                                        time_stopping = gravity_turn_final_time + coasting_time,
                                        plot_bool = True)