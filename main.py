import math
import numpy as np


landing_prop_percentage_stage_1 = 0.98
landing_prop_percentage_stage_2 = 0.98

# Stage 1 : Super Heavy
m_stage_1 = 3675e3 # kg
m_prop_1 = 3400e3 # kg
m_strc_1 = 275e3 # kg

# Stage 2 : Starship
m_stage_2 = 1600e3 # kg
m_prop_2 = 1500e3 # kg
m_strc_2 = 100e3 # kg

# Stage 3 : Payload
m_payload = 100e3 # kg

# Convert to useful masses
m_initial = m_stage_1 + m_stage_2 + m_payload
m_stage_1_burn_out = m_initial - m_prop_1 * landing_prop_percentage_stage_1
m_stage_1_separation = m_stage_1_burn_out - m_strc_1
m_stage_2_burn_out = m_stage_1_separation - m_prop_2 * landing_prop_percentage_stage_2

# Rocket parameters
D_rocket = 9 # [m]
S_rocket = np.pi * (D_rocket/2)**2 # [m^2]
from drag_coeff import compile_drag_coefficient_func
get_drag_coefficient_func_stage_1 = compile_drag_coefficient_func(alpha = 5)
get_drag_coefficient_func_stage_2 = compile_drag_coefficient_func(alpha = 5)

# Engine parameters
Isp_stage_1 = 350 # [s]
Isp_stage_2 = 380 # [s]

v_ex_stage_1 = Isp_stage_1 * 9.81 # [m/s]
v_ex_stage_2 = Isp_stage_2 * 9.81 # [m/s]

T_engine_stage_1 = 2745e3 # [N]
T_engine_stage_2 = 2000e3 # [N]
n_engine_stage_1 = 33
n_engine_stage_2 = 6

T_max_stage_1 = T_engine_stage_1 * n_engine_stage_1 # [N]
T_max_stage_2 = T_engine_stage_2 * n_engine_stage_2 # [N]

m_dot_stage_1 = T_max_stage_1 / v_ex_stage_1 # [kg/s]
m_dot_stage_2 = T_max_stage_2 / v_ex_stage_2 # [kg/s]

t_burn_stage_1 = m_prop_1 / m_dot_stage_1 # [s]
t_burn_stage_2 = m_prop_2 / m_dot_stage_2 # [s]

nozzle_exit_area = 1.326 # [m^2]
nozzle_exit_pressure_stage_1 = 100000 # [Pa]
nozzle_exit_pressure_stage_2 = 100000 # [Pa]

### Orbital parameters ###
h_vertical_rising = 100 # [m]
h_gravity_turn = 100e3 # [m] but go to max 100e3

kick_angle = math.radians(-1.21) # [rad] : so ends at roughly 45 degrees flight path angle.

throttle_gravity_turn = 0.95
h_throttle_gt_0 = 5000 # [m]
h_throttle_gt_1 = 20000 # [m]

t_coast_endo = 5 # [s]

from trans import plot_eci_to_local_xyz
### PERFORM VERTICAL RISING ###

from vertical_rising import endo_atmospheric_vertical_rising

times, states, final_state, unit_east_vector = endo_atmospheric_vertical_rising(initial_mass = m_initial,
                                                                                target_altitude = h_vertical_rising,
                                                                                minimum_mass = m_stage_1_burn_out,
                                                                                mass_flow_endo = m_dot_stage_1,
                                                                                specfic_impulse_vacuum = Isp_stage_1,
                                                                                get_drag_coefficient_func = get_drag_coefficient_func_stage_1,
                                                                                frontal_area = S_rocket,
                                                                                nozzle_exit_area = nozzle_exit_area,
                                                                                nozzle_exit_pressure = nozzle_exit_pressure_stage_1,
                                                                                number_of_engines = n_engine_stage_1)

earth_rotation_angle, final_state_local = plot_eci_to_local_xyz(states,
                          times,
                          0,
                          0,
                          None,
                          'vertical_rising')
final_time_previous_phase = times[-1]
from gravity_turn import endo_atmospheric_gravity_turn

times, states, final_state, max_dynamic_pressure = endo_atmospheric_gravity_turn(vertical_rising_final_state = final_state,
                                                                                 kick_angle = kick_angle,
                                                                                 unit_east_vector = unit_east_vector,
                                                                                 t_start = times[-1],
                                                                                 target_altitude = h_gravity_turn,
                                                                                 minimum_mass = m_stage_1_burn_out,
                                                                                 mass_flow_endo = m_dot_stage_1,
                                                                                 specific_impulse_vacuum = Isp_stage_1,
                                                                                 get_drag_coefficient_func = get_drag_coefficient_func_stage_1,
                                                                                 frontal_area = S_rocket,
                                                                                 nozzle_exit_area = nozzle_exit_area,
                                                                                 nozzle_exit_pressure = nozzle_exit_pressure_stage_1,
                                                                                 number_of_engines = n_engine_stage_1,
                                                                                 thrust_throttle = throttle_gravity_turn,
                                                                                 thrust_altitudes = (h_throttle_gt_0, h_throttle_gt_1))

earth_rotation_angle, final_state_local = plot_eci_to_local_xyz(states,
                          times,
                          earth_rotation_angle,
                          final_time_previous_phase,
                          final_state_local,
                          'gravity_turn')
final_time_previous_phase = times[-1]

print(f'final state local gravity turn: {final_state_local}')
### Endo coasting ###

from endo_coasting import endo_coasting

times, states, final_state, coasting_times, coasting_states = endo_coasting(previous_times = times,
                                                                          previous_states = states,
                                                                          rocket_mass = m_stage_1_separation,
                                                                          coasting_time = t_coast_endo,
                                                                          frontal_area = S_rocket,
                                                                          get_drag_coefficient_func = get_drag_coefficient_func_stage_2)


earth_rotation_angle, final_state_local = plot_eci_to_local_xyz(coasting_states,
                          coasting_times,
                          earth_rotation_angle,
                          final_time_previous_phase,
                          final_state_local,
                          'endo_coasting')
final_time_previous_phase = coasting_times[-1]

print(f'final state local coasting: {final_state_local}')

### Exo propelled ###

from exo_dyn_opt import exo_propelled

states_exo, times_exo, states, times, final_state_exo = exo_propelled(initial_state = final_state,
                                                      previous_times = times,
                                                      previous_states = states,
                                                      v_exhaust_stage_2 = v_ex_stage_2,
                                                      mass_flow_per_engine_stage_2 = m_dot_stage_2,
                                                      number_of_engines_stage_2 = n_engine_stage_2,
                                                      propellant_mass_stage_2_ascent = m_prop_2,
                                                      time_scale = 100,
                                                      mass_burnout = m_stage_2_burn_out,
                                                      semi_major_axis = 6378137 + 200e3,
                                                      number_of_iterations = 1000,
                                                      mission = 'LEO',
                                                      print_bool = True)

earth_rotation_angle, final_state_local = plot_eci_to_local_xyz(states_exo,
                          times_exo,
                          earth_rotation_angle,
                          final_time_previous_phase,
                          final_state_local,
                          'exo_propelled')
final_time_previous_phase = times_exo[-1]

### Exo coasting ###

from exo_coasting import exo_atmosphere_coasting_to_orbit

times_exo_coasting, states_exo_coasting, final_state_exo_coasting = exo_atmosphere_coasting_to_orbit(t_start = times[-1],
                                                                                                      initial_state = final_state_exo,
                                                                                                      target_altitude = 200e3)

earth_rotation_angle, final_state_local = plot_eci_to_local_xyz(states_exo_coasting,
                          times_exo_coasting,
                          earth_rotation_angle,
                          final_time_previous_phase,
                          final_state_local,
                          'exo_coasting')
final_time_previous_phase = times_exo[-1]