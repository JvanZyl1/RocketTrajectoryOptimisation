import csv
import numpy as np
from src.TrajectoryGeneration.Transformations import plot_eci_to_local_xyz
from src.envs.utils.aerodynamic_coefficients import compile_drag_coefficient_func
from src.TrajectoryGeneration.flight_phases.gravity_turn import endo_atmospheric_gravity_turn
from src.TrajectoryGeneration.flight_phases.vertical_rising import endo_atmospheric_vertical_rising

R_e = 6378137.0

class AscentTrajectoryOptimiser:
    def __init__(self):
        # Orbit pre-set parameters
        self.h_vertical_rising = 100    # [m]
        self.h_target = 200e3           # [km]
        self.a = R_e + self.h_vertical_rising
        self.alpha_drag_deg = 2

        # Constants
        self.CD_func = compile_drag_coefficient_func(alpha_degrees=self.alpha_drag_deg)

        # Read sizing results
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]

        self.m_initial = float(sizing_results['Initial mass (subrocket 0)']) * 1000
        self.m_subrocket_subrocket_1_burn_out_ascent = float(sizing_results['Ascent burnout mass (subrocket 0)']) * 1000
        self.frontal_area = float(sizing_results['Rocket frontal area'])
        self.nozzle_exit_area = float(sizing_results['Nozzle exit area'])
        self.nozzle_exit_pressure_stage_1 = float(sizing_results['Nozzle exit pressure stage 1'])
        self.n_engine_stage_1 = int(sizing_results['Number of engines stage 1'])
        self.m_dot_stage_1 = (float(sizing_results['Maximum thrust stage 1'])*1e6)/(float(sizing_results['Exhaust velocity stage 1']))
        self.Isp_stage_1 = float(sizing_results['Exhaust velocity stage 1'])/9.81

        self.m_subrocket_subrocket_2_initial = float(sizing_results['Mass at stage separation (subrocket 1)'])*1000
        self.m_subrocket_stage_2_burn_out = float(sizing_results['Ascent burnout mass (subrocket 1)'])*1000
        self.m_payload = float(sizing_results['Payload mass'])*1000
        self.n_engine_stage_2 = float(sizing_results[''])
        self.m_dot_stage_2 = (float(sizing_results['Maximum thrust stage 2'])*1e6)/(float(sizing_results['Exhaust velocity stage 2']))
        self.Isp_stage_2 = float(sizing_results['Exhaust velocity stage 2'])/9.81

    def reset(self):
        self.times = []
        self.states = []
        self.final_time_previous_phase = 0.0
        self.earth_rotation_angle = 0.0
        self.final_state_local = None
        self.final_state = []
        self.unit_east_vector = []

    def vertical_rising(self):
        self.times, self.states, self.final_state, self.unit_east_vector = endo_atmospheric_vertical_rising(initial_mass = self.m_initial,
                                                                                        target_altitude = self.h_vertical_rising,
                                                                                        minimum_mass = self.m_subrocket_subrocket_1_burn_out_ascent,
                                                                                        mass_flow_endo = self.m_dot_stage_1,
                                                                                        specfic_impulse_vacuum = self.Isp_stage_1,
                                                                                        get_drag_coefficient_func = self.CD_func,
                                                                                        frontal_area = self.frontal_area,
                                                                                        nozzle_exit_area = self.nozzle_exit_area,
                                                                                        nozzle_exit_pressure = self.nozzle_exit_pressure_stage_1,
                                                                                        number_of_engines = self.n_engine_stage_1)

        self.earth_rotation_angle, self.final_state_local = plot_eci_to_local_xyz(self.states, self.times, self.earth_rotation_angle, self.final_time_previous_phase, self.final_state_local, 'vertical_rising')
        self.final_time_previous_phase = self.times[-1]

    def gravity_turn(self, kick_angle : float):
        times_gt, states_gt, final_state_gt, max_dynamic_pressure = endo_atmospheric_gravity_turn(vertical_rising_final_state = self.final_state,
                                                                                    kick_angle = kick_angle,
                                                                                    unit_east_vector = self.unit_east_vector,
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
    
        states = np.concatenate((states, states_gt), axis=1)
        times = np.concatenate((times, times_gt))

        earth_rotation_angle, final_state_local, flight_path_angle, states_local = plot_eci_to_local_xyz(states,
                                times,
                                earth_rotation_angle,
                                final_time_previous_phase,
                                final_state_local,
                                'gravity_turn',
                                return_final_gamma_deg = True)
        final_time_previous_phase = times[-1]


        