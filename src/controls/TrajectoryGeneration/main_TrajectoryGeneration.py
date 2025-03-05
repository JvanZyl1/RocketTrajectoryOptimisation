from src.controls.TrajectoryGeneration.Transformations import plot_eci_to_local_xyz, calculate_flight_path_angles
from src.controls.TrajectoryGeneration.vertical_rising import endo_atmospheric_vertical_rising
from src.controls.TrajectoryGeneration.gravity_turn import endo_atmospheric_gravity_turn






def endo_trajectory_generation_test(kick_angle,
                                    throttle_gravity_turn,
                                    m_initial,
                                    m_stage_1_burn_out,
                                    S_rocket,
                                    nozzle_exit_area,
                                    nozzle_exit_pressure_stage_1,
                                    n_engine_stage_1,
                                    m_dot_stage_1,
                                    Isp_stage_1,
                                    get_drag_coefficient_func_stage_1):
    ### Orbital parameters ###
    h_vertical_rising = 100 # [m]
    h_gravity_turn = 100e3 # [m] but go to max 100e3
    h_throttle_gt_0 = 5000 # [m]
    h_throttle_gt_1 = 20000 # [m]

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

    earth_rotation_angle, final_state_local, flight_path_angle = plot_eci_to_local_xyz(states,
                            times,
                            earth_rotation_angle,
                            final_time_previous_phase,
                            final_state_local,
                            'gravity_turn',
                            return_final_gamma_deg = True)
    final_time_previous_phase = times[-1]

    r_up = final_state_local[0]

    return r_up, flight_path_angle, max_dynamic_pressure, times, states


    
