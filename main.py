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
# Display outputs
if print_bool:
      print("Initial Mass [kg]:", initial_mass)
      print("Subrocket Masses [kg]:", sub_stage_masses) # Final is payload
      print("Stages Masses [kg]:", stage_masses)
      print("Structural Masses [kg]:", structural_masses)
      print("Propellant Masses [kg]:", propellant_masses)
      print("Delta-V Required [m/s]:", delta_v_required)
      print("Stage Delta-V [m/s]:", delta_v_required_stages)
      print("Payload Ratios:", payload_ratios)
      print("Mass Ratios:", mass_ratios)

from functions.params import maximum_accelerations, g0, mass_fairing

stage_properties_dict = compute_stage_properties(initial_mass,
                                                 stage_masses,
                                                 propellant_masses,
                                                 structural_masses,
                                                 specific_impulses_vacuum,
                                                 g0,
                                                 maximum_accelerations,
                                                 mass_fairing)


####
import json
from functions.utils import convert_ndarray

# Assuming all your variables are defined as per your snippet
data_to_save = {
    'initial_mass': initial_mass,
    'sub_stage_masses': convert_ndarray(sub_stage_masses),
    'stage_masses': convert_ndarray(stage_masses),
    'structural_masses': convert_ndarray(structural_masses),
    'propellant_masses': convert_ndarray(propellant_masses),
    'delta_v_required': delta_v_required,
    'delta_v_required_stages': convert_ndarray(delta_v_required_stages),
    'payload_ratios': convert_ndarray(payload_ratios),
    'mass_ratios': convert_ndarray(mass_ratios),
    'stage_properties_dict': convert_ndarray(stage_properties_dict)
}
# Specify the filename
filename = 'data/rocket_stage_data.json'

# Write the dictionary to a JSON file
with open(filename, 'w') as json_file:
    json.dump(data_to_save, json_file, indent=4)  # indent=4 for readability

print(f"Data successfully written to {filename}")
###


# Display outputs
if print_bool:
      print("\nStage Properties:")
      print(stage_properties_dict)

# Load and display initial conditions
from functions.params import initial_conditions_dictionary, target_altitude_vertical_rising
if print_bool:
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
if print_bool:
      print(f'\nInitial State for Vertical Rising: {initial_state_vertical_rising}')
from functions.endo_atmosphere_vertical_rising import endo_atmospheric_vertical_rising
vertical_rising_time, vertical_rising_states, \
      vertical_rising_final_state = endo_atmospheric_vertical_rising(initial_state_vertical_rising,
                                    target_altitude_vertical_rising,
                                    stage_properties_dict['burn_out_masses'][0],
                                    stage_properties_dict['mass_flow_rates'][0],
                                    plot_bool = False)

from functions.params import w_earth, kick_angle, unit_east_vector
if print_bool:
      print(f'vertical_rising_final_state: {vertical_rising_final_state}')
##### DELTA-V CALCULATION #####
from functions.losses_calculator import losses_over_states
vertical_rising_losses = losses_over_states(vertical_rising_states,
                                            vertical_rising_time,
                                            endo_atmosphere_bool=True)
#####
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
if print_bool:
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
print(f'initial_state_GT: {initial_state_GT}')

if print_bool:
      print(f'initial_state_GT: {initial_state_GT}')

# Implement gravity turn with a gradual kick-angle change.
from functions.endo_atmosphere_gravity_turn import endo_atmospheric_gravity_turn
from functions.params import target_altitude_gravity_turn, R_earth

gravity_turn_time, gravity_turn_states, \
      gravity_turn_final_state = endo_atmospheric_gravity_turn(t_start = vertical_rising_final_time,
                                    initial_state = initial_state_GT,
                                    target_altitude = target_altitude_gravity_turn,
                                    minimum_mass=stage_properties_dict['burn_out_masses'][0],
                                    mass_flow_endo=stage_properties_dict['mass_flow_rates'][0],
                                    plot_bool = False)
##### DELTA-V CALCULATION #####
gravity_turn_losses = losses_over_states(vertical_rising_states,
                                            vertical_rising_time,
                                            endo_atmosphere_bool=True)
#####
gravity_turn_final_position_vector = gravity_turn_final_state[:3]
gravity_turn_final_velocity_vector = gravity_turn_final_state[3:6]
gravity_turn_final_mass = gravity_turn_final_state[6]
gravity_turn_final_time = gravity_turn_time[-1]

gravity_turn_final_altitude = np.linalg.norm(gravity_turn_final_position_vector) - R_earth

if print_bool:
      print(f'gravity_turn_final_state: {gravity_turn_final_state}'
            f'\n altitude: {gravity_turn_final_altitude}'
            f'\n target altitude: {target_altitude_gravity_turn}'
            f'\n minimum mass: {stage_properties_dict["burn_out_masses"][0]}')

### Coasting procedure
# The rocket first stage has departed.
# The fairing along with it.
mass_coasting = sub_stage_masses[1] - mass_fairing # Mass of the second stage & payload - fairing [kg]
if print_bool:
      print(f'mass_coasting: {mass_coasting}')
coasting_first_state = np.concatenate((gravity_turn_final_position_vector,
                                          gravity_turn_final_velocity_vector,
                                          [mass_coasting]))

from functions.endo_atmosphere_coasting import endo_atmosphere_coasting
from functions.params import coasting_time

coasting_time, coasting_states, \
        coasting_final_state = endo_atmosphere_coasting(t_start = gravity_turn_final_time,
                                        initial_state = coasting_first_state,
                                        time_stopping = gravity_turn_final_time + coasting_time,
                                        plot_bool = False)

# Optimise exo-atmospheric trajectory.
coasting_final_position_vector = coasting_final_state[:3]
coasting_final_velocity_vector = coasting_final_state[3:6]
coasting_final_mass = coasting_final_state[6]
coasting_final_altitude = np.cross(coasting_final_position_vector, coasting_final_velocity_vector)
coasting_final_time = coasting_time[-1]
if print_bool:
      print(f'Gravity turn final state: {gravity_turn_final_state}')
      print(f'Coasting final state: {coasting_final_state}')


from functions.exo_atmos_opt import ExoAtmosphericPropelledOptimisation
from functions.params import exo_atmoshere_target_altitude_propelled, minimum_delta_v_adjustments_exo

exo_atmos_opt = ExoAtmosphericPropelledOptimisation(
      Isp = specific_impulses_vacuum[1],
      semi_major_axis = semi_major_axis,
      initial_state = coasting_final_state,
      structural_mass = structural_masses[1],
      mass_flow_exo = stage_properties_dict['mass_flow_rates'][1],
      mass_payload = payload_mass,
      burn_time_exo_stage = stage_properties_dict["burn_times"][1],
      max_altitude = exo_atmoshere_target_altitude_propelled,
      minimum_delta_v_adjustments = minimum_delta_v_adjustments_exo,
      print_bool = True,
      number_of_iterations = 550) # Many more for true optimal solution, but 200 gives somewhere which kind of works.

exo_propelled_optimised_variables = exo_atmos_opt.optimise() #[burn_time, prU, pvU]
if print_bool:
      print(f'Optimised variables: {exo_propelled_optimised_variables}')

if print_bool:
      print(f'propellant_burn_time: {exo_propelled_optimised_variables[0]} vs. 171'
            f'\n prU: {exo_propelled_optimised_variables[1:4]} vs. [-0.004, -0.021, -0.0435]'
            f'\n pvU: {exo_propelled_optimised_variables[4:7]} vs. [1, -0.2716, 0.7936]')

from functions.exo_atmopshere_propelled import exo_atmosphere_propelled
#sprint(f'Validation :exo_propelled_optimised_variables = {[171, -0.004, -0.021, -0.0435, 1, -0.2716, 0.7936]}')

exo_propelled_final_state_augmented, exo_propelled_augmented_states, \
      exo_propelled_times = exo_atmosphere_propelled(initial_state = coasting_final_state,
                                    optimisation_parameters = exo_propelled_optimised_variables,
                                    t_start = coasting_final_time,
                                    mass_flow_exo = stage_properties_dict['mass_flow_rates'][1],
                                    plot_bool = False)

# Coasting to final orbit.
from functions.params import altitude_orbit
from functions.exo_atmosphere_coasting_to_orbit import exo_atmosphere_coasting_to_orbit
exo_coasting_time_start = exo_propelled_times[-1]
exo_coasting_initial_state = exo_propelled_final_state_augmented[:7] #[r, v, m]

exo_coasting_times, exo_coasting_states, \
            exo_coasting_final_state = exo_atmosphere_coasting_to_orbit(t_start = exo_coasting_time_start,
                                            initial_state = exo_coasting_initial_state,
                                            target_altitude = altitude_orbit,
                                            plot_bool = False)
# Circularise final orbit.
from functions.exo_circular_final_orbit import final_orbit_maneuver
final_orbit_time_start = exo_coasting_times[-1]
final_orbit_initial_state = exo_coasting_final_state # [r, v, m]

final_orbit_times, final_orbit_states, \
            final_orbit_final_state = final_orbit_maneuver(initial_state = final_orbit_initial_state,
                                            altitude_orbit = altitude_orbit,
                                            t_start = final_orbit_time_start,
                                            plot_bool = False)

# Collate trajectory times
trajectory_times = [vertical_rising_time,
                    gravity_turn_time[1:],
                    coasting_time[1:],
                    exo_propelled_times[1:],
                    exo_coasting_times[1:],
                    final_orbit_times[1:]]
trajectory_times = np.concatenate(trajectory_times)
# Collate trajectory states
exo_propelled_states_non_augmented = exo_propelled_augmented_states[:7,:] # [r, v, m]

trajectory_states = [vertical_rising_states,
                        gravity_turn_states[:,1:],
                        coasting_states[:,1:],
                        exo_propelled_states_non_augmented[:,1:],
                        exo_coasting_states[:,1:],
                        final_orbit_states[:,1:]]
trajectory_states = np.concatenate(trajectory_states, axis=1)

# Now without orbit
trajectory_states_no_orbit = [vertical_rising_states,
                        gravity_turn_states[:,1:],
                        coasting_states[:,1:],
                        exo_propelled_states_non_augmented[:,1:],
                        exo_coasting_states[:,1:]]
trajectory_states_no_orbit = np.concatenate(trajectory_states_no_orbit, axis=1)

trajectory_times_no_orbit = [vertical_rising_time,
                    gravity_turn_time[1:],
                    coasting_time[1:],
                    exo_propelled_times[1:],
                    exo_coasting_times[1:]]
trajectory_times_no_orbit = np.concatenate(trajectory_times_no_orbit)

# Plot trajectories
from functions.plot_final_trajectory import final_orbit_plotter
final_orbit_plotter(trajectory_states,
                        trajectory_times,
                        plot_bool=False,
                        full_orbit=True)

final_orbit_plotter(trajectory_states_no_orbit,
                        trajectory_times_no_orbit,
                        plot_bool=False,
                        full_orbit=False)


# Flip maneuver
# Boostback burn
# Re-entry burn
# Landing burn