import numpy as np
import math
from TrajectoryGeneration.drag_coeff import compile_drag_coefficient_func
from TrajectoryGeneration.main_TrajectoryGeneration import endo_trajectory_generation_test
from RocketSizing.staging import staging_reusable_rocketry
from RocketSizing.rocket_radius_calc import new_radius_func
from RocketSizing.rocket_dimensions import rocket_dimensions
R_earth = 6378137 # [m]

class create_rocket_configuration:
    def __init__(self,
                 delta_v_loss_ascent : np.array,
                 delta_v_descent : np.array):
        
        # Constants
        self.max_dynamic_pressure = 70000 # [Pa]
        
        # Always LEO
        self.m_payload = 100e3 # kg
        self.semi_major_axis = R_earth + 200e3 # [m]
        self.number_of_stages = 2

        # Load Raptor constants
        self.load_raptor_constants()

        # stage rocket
        self.delta_v_loss_ascent = delta_v_loss_ascent
        self.delta_v_descent = delta_v_descent
        self.stage_rocket(delta_v_loss_ascent,
                          delta_v_descent)
        self.number_of_engines()

        # Test trajectory generation
        self.mock_times, self.mock_states = self.test_trajectory_generation()

        # Inertia calculator
        self.inertia_calculator()



        

    def load_raptor_constants(self):
        self.Isp_stage_1 = 350 # [s]
        self.Isp_stage_2 = 380 # [s]

        self.v_ex_stage_1 = self.Isp_stage_1 * 9.81 # [m/s]
        self.v_ex_stage_2 = self.Isp_stage_2 * 9.81 # [m/s]

        self.T_engine_stage_1 = 2745e3 # [N]
        self.T_engine_stage_2 = 2000e3 # [N]

        self.nozzle_exit_area = 1.326 # [m^2]
        self.nozzle_exit_pressure_stage_1 = 100000 # [Pa] - TODO: Check if this is correct
        self.nozzle_exit_pressure_stage_2 = 100000 # [Pa] - TODO: Check if this is correct

    def stage_rocket(self,
                      delta_v_loss_ascent : np.array,
                      delta_v_descent : np.array):
        # Super heavy : https://en.wikipedia.org/wiki/SpaceX_Super_Heavy
        total_mass_super_heavy = 3675           # [t] : Total mass of the super heavy := mass propellant + mass dry
        propellant_mass_super_heavy = 3400      # [t] : Propellant mass of the super heavy
        structural_mass_super_heavy = total_mass_super_heavy - propellant_mass_super_heavy
        structural_coefficient_super_heavy = structural_mass_super_heavy / propellant_mass_super_heavy

        # Starship : https://en.wikipedia.org/wiki/SpaceX_Starship_(spacecraft)
        structural_mass_starship = 100           # [t] : Structural mass of the starship
        propellant_mass_starship = 1500          # [t] : Propellant mass of the starship
        structural_coefficient_starship = structural_mass_starship/ propellant_mass_starship

        stage_dict = staging_reusable_rocketry(self.semi_major_axis,
                                               self.m_payload,
                                               delta_v_loss_ascent,
                                               delta_v_descent,
                                               self.number_of_stages,
                                               np.array([self.v_ex_stage_1, self.v_ex_stage_2]),
                                               np.array([structural_coefficient_super_heavy, structural_coefficient_starship]),
                                               np.array([0]),
                                               False)
        
        self.m_initial = stage_dict['initial_mass']
        self.m_stage_1_ascent_burnout = stage_dict['mass_at_stage_1_ascent_burnout']
        self.m_prop_1 = stage_dict['propellant_mass_stage_1_ascent'] + stage_dict['propellant_mass_stage_1_descent']
        self.m_prop_2 = stage_dict['propellant_mass_stage_2_ascent']
        self.m_stage_1 = stage_dict['structural_mass_stage_1_ascent'] + stage_dict['propellant_mass_stage_1_ascent']
        self.m_stage_2 = stage_dict['structural_mass_stage_2_ascent'] + stage_dict['propellant_mass_stage_2_ascent']

        with open('results/env_values.txt', 'w') as file:
            file.write(f"Structural mass stage 1 (ascent) [ton]: {stage_dict['structural_mass_stage_1_ascent']/1e3}\n")
            file.write(f"Structural mass stage 2 (ascent) [ton]: {stage_dict['structural_mass_stage_2_ascent']/1e3}\n")
            file.write(f"Propellant mass stage 1 (ascent) [ton]: {stage_dict['propellant_mass_stage_1_ascent']/1e3}\n")
            file.write(f"Propellant mass stage 2 (ascent) [ton]: {stage_dict['propellant_mass_stage_2_ascent']/1e3}\n")
            file.write(f"Structural mass stage 1 (descent) [ton]: {stage_dict['structural_mass_stage_1_descent']/1e3}\n")
            file.write(f"Structural mass stage 2 (descent) [ton]: - \n")
            file.write(f"Propellant mass stage 1 (descent) [ton]: {stage_dict['propellant_mass_stage_1_descent']/1e3} \n")
            file.write(f"Propellant mass stage 2 (descent) [ton]: - \n")
            file.write(f"Actual structural mass stage 1 [ton]: {self.m_stage_1/1e3}\n")
            file.write(f"Actual structural mass stage 2 [ton]: {self.m_stage_2/1e3}\n")
            file.write(f"Actual propellant mass stage 1 [ton]: {self.m_prop_1/1e3}\n")
            file.write(f"Actual propellant mass stage 2 [ton]: {self.m_prop_2/1e3}\n")
            file.write(f"Initial mass (subrocket 0) [ton]: {self.m_initial/1e3}\n")
            file.write(f"Initial mass (subrocket 1) [ton]: {stage_dict['mass_of_rocket_at_stage_1_separation']/1e3}\n")
            file.write(f"Ascent burnout mass (subrocket 0) [ton]: {self.m_stage_1_ascent_burnout/1e3}\n")
            file.write(f"Ascent burnout mass (subrocket 1) [ton]: {stage_dict['mass_at_stage_2_ascent_burnout']/1e3}\n")
            file.write(f"Mass at stage separation (subrocket 0) [ton]: {stage_dict['mass_of_stage_1_at_separation']/1e3}\n")
            file.write(f"Mass at stage separation (subrocket 1) [ton]: {stage_dict['mass_of_rocket_at_stage_1_separation']/1e3}\n")
            file.write(f"Payload mass [ton]: {self.m_payload/1e3}\n")

    def number_of_engines(self, TWR_stage_1 = 2.51, TWR_stage_2 = 0.76):
        thrust_req_stage_1 = self.m_stage_1 * 9.81 * TWR_stage_1
        thrust_req_stage_2 = self.m_stage_2 * 9.81 * TWR_stage_2

        self.n_engine_stage_1 = math.ceil(thrust_req_stage_1 / self.T_engine_stage_1) + 3
        T_max_stage_1 = self.T_engine_stage_1 * self.n_engine_stage_1 # [N]

        self.n_engine_stage_2 = math.ceil(thrust_req_stage_2 / self.T_engine_stage_2)
        T_max_stage_2 = self.T_engine_stage_2 * self.n_engine_stage_2 # [N]

        self.m_dot_stage_1 = T_max_stage_1 / self.v_ex_stage_1 # [kg/s]
        self.m_dot_stage_2 = T_max_stage_2 / self.v_ex_stage_2 # [kg/s]

        self.t_burn_stage_1 = self.m_prop_1 / self.m_dot_stage_1 # [s]
        self.t_burn_stage_2 = self.m_prop_2 / self.m_dot_stage_2 # [s]

        self.radius_rocket, self.S_rocket, number_of_engines_per_ring = new_radius_func(self.n_engine_stage_1)
        self.stage_1_n_gimballed = number_of_engines_per_ring[1] + number_of_engines_per_ring[2]

        with open('results/env_values.txt', 'a') as file:
            file.write(f'Number of engines stage 1: {self.n_engine_stage_1}\n')
            file.write(f'Number of engines stage 2: {self.n_engine_stage_2}\n')
            file.write(f'Number of engines per ring stage 1: {number_of_engines_per_ring}\n')
            file.write(f'Rocket Radius [m]: {self.radius_rocket}\n')
            file.write(f'Rocket frontal area [m^2]: {self.S_rocket}\n')
            file.write(f'Maximum thrust stage 1 [MN]: {T_max_stage_1/1e6}\n')
            file.write(f'Maximum thrust stage 2 [MN]: {T_max_stage_2/1e6}\n')
            file.write(f'Burn time stage 1 [s]: {self.t_burn_stage_1}\n')
            file.write(f'Burn time stage 2 [s]: {self.t_burn_stage_2}\n')


    def test_trajectory_generation(self, TWR_base = 2.51):
        get_drag_coefficient_func_stage_1 = compile_drag_coefficient_func(alpha = 5)

        endo_trajectory_lambda = lambda kick_angle, throttle_gravity_turn : endo_trajectory_generation_test(kick_angle,
                                    throttle_gravity_turn,
                                    self.m_initial,
                                    self.m_stage_1_ascent_burnout,
                                    self.S_rocket,
                                    self.nozzle_exit_area,
                                    self.nozzle_exit_pressure_stage_1,
                                    self.n_engine_stage_1,
                                    self.m_dot_stage_1,
                                    self.Isp_stage_1,
                                    get_drag_coefficient_func_stage_1)
        
        # Iterate throttle and kick to generate mock ascent trajectory.
        kick_angle_abs_range = np.linspace(-math.radians(0.4), -math.radians(1), 20)
        throttle_range = np.linspace(1, 0.7, 5)

        for kick_angle in kick_angle_abs_range:
            for throttle in throttle_range:
                r_up, flight_path_angle, max_dynamic_pressure, times, states = endo_trajectory_lambda(kick_angle, throttle)
                print(f'Testing kick angle: {math.degrees(kick_angle)} and throttle: {throttle}, Reached altitude: {r_up} m at Flight path angle: {flight_path_angle} deg')
                if r_up < 50e3:
                    print(f'Does not go high enough, only reached {r_up} m. Resizing rocket by adding more engines or increasing propellant.')
                    # Adjust the rocket configuration
                    TWR_base -= 0.05
                    self.number_of_engines(TWR_base)
                    # Restart the loop
                    return self.test_trajectory_generation(TWR_base)
                elif max_dynamic_pressure > self.max_dynamic_pressure:
                    print(f'Max dynamic pressure too high, {max_dynamic_pressure/1000} kPa. Reducing throttle.')
                elif flight_path_angle > 50:
                    print(f'Flight path angle too high, {flight_path_angle} deg. Increasing kick angle.')
                    break
                elif flight_path_angle < 40:
                    print(f'Flight path angle overshot now too low, {math.degrees(flight_path_angle)} deg. STOP CODE and make a finer mesh on kick angle.')
                    raise ValueError('Flight path angle too low')
                else:
                    print(f'Altitude reached, Dynamic pressure maintained, and flight path angle is good. This is a good configuration.')

                    with open('results/env_values.txt', 'a') as file:
                        file.write(f'Maximum dynamic pressure allowed [kPa]: {self.max_dynamic_pressure/1000}\n')
                        file.write(f'Maximum dynamic pressure reached [kPa]: {max_dynamic_pressure/1000}\n')
                        file.write(f'Target altitude vertical rising [km]: {0.1}\n')
                        file.write(f'Target altitude gravity turn [km]: {50}\n')
                        file.write(f'Kick angle [deg]: {math.degrees(kick_angle)}\n')
                        file.write(f'Throttle: {throttle}\n')
                        file.write(f'Flight path angle reached in gravity turn [deg]: {flight_path_angle}\n')
                        file.write(f'Start of gravity turn throttle [km]: {5}\n')
                        file.write(f'End of gravity turn throttle [km]: {20}\n')
                    return times, states
                
    def inertia_calculator(self):
        # Create an instance of rocket_dimensions with the required arguments
        rocket_dimensions_instance = rocket_dimensions(
            rocket_radius=self.radius_rocket,
            propellant_masses=[self.m_prop_1, self.m_prop_2],
            structural_masses=[self.m_stage_1, self.m_stage_2],
            payload_mass=self.m_payload,
            number_of_engines=[self.n_engine_stage_1, self.n_engine_stage_2]
        )
        
        # Call the instance to get the required values
        self.x_cog_inertia_subrocket_0_lambda, self.x_cog_inertia_subrocket_1_lambda, self.lengths, self.x_cog_payload = rocket_dimensions_instance()


if __name__ == '__main__':
    delta_v_loss_ascent = np.array([400, 50])
    delta_v_descent = np.array([50, 0])

    rocket_config = create_rocket_configuration(delta_v_loss_ascent, delta_v_descent)