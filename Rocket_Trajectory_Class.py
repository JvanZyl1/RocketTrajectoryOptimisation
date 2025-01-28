import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from functions.staging import staging_expendable, compute_stage_properties


physical_constants = {
    'mu': 398602 * 1e9,                                 # Gravitational parameter [m^3/s^2]
    'R_earth': 6378137,                                 # Earth radius [m]
    'w_earth': np.array([0, 0, 2 * np.pi / 86164]),     # Earth angular velocity [rad/s]
    'g0': 9.80665,                                      # Gravity constant on Earth [m/s^2]
    'scale_height_endo': 8500,                          # Scale height for endo-atmospheric model [m]
    'rho0': 1.225,                                      # Sea level density [kg/m^3]
    'M_earth': 5.972e24,                                # Earth mass [kg]
    'G': 6.67430e-11                                    # Gravitational constant [m^3/kg/s^2]
}

mission_requirements = {
    'payload_mass': 300,                                # Payload mass [kg]
    'mass_fairing': 100,                                # Fairing mass [kg]
    'altitude_orbit': 700000,                           # Orbit altitude [m]
    'max_first_stage_g': 7,                             # Maximum first stage acceleration [g0]
    'max_second_stage_g': 6,                            # Maximum second stage acceleration [g0]
    'number_of_stages': 2,                              # Number of stages [int]
    'structural_coefficients': [0.10, 0.13],            # Structural Coefficient [float]
    'specific_impulses_vacuum': [300, 320],             # Vacuum Specific Impulse [s]
    'launch_site_latitude': 5.2 * np.pi / 180           # Kourou latitude [rad] - launch altitude
}

design_parameters = {
    'aerodynamic_area': 1,                              # Reference aerodynamic area [m^2] {endo-phase ascent}
    'nozzle_exit_area': 0.3,                            # Exhaust nozzle area [m^2]
    'nozzle_exit_pressure': 40000,                      # Nozzle exit pressure [Pa]
    'mach_number_array': np.array([0.2, 0.5, 0.8, 1.2, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]),  # Mach number [-]
    'cd_array': np.array([0.27, 0.26, 0.25, 0.5, 0.46, 0.44, 0.41, 0.39, 0.37, 0.35, 0.33, 0.3, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21])  # Drag coefficient [-]
}

mission_profile = {
    'target_altitude_vertical_rising': 100.0,           # target altitude [m]
    'kick_angle': math.radians(5),                      # kick angle [deg]
    'target_altitude_gravity_turn': 1160000.0,          # target altitude [m], maximum altitude for gravity turn
    'coasting_time': 5.0,                               # coasting time [s]
    'exo_atmoshere_target_altitude_propelled': 100000.0, # target altitude [m]
    'minimum_delta_v_adjustments_exo': 200.0            # minimum delta v adjustments [m/s]
}

drag_coefficient_function = interp1d(
    design_parameters['mach_number_array'],
    design_parameters['cd_array'],
    kind='linear',  # Linear interpolation
    fill_value='extrapolate'  # Allow extrapolation for Mach numbers outside the range
)

def get_drag_coefficient(mach_number):
    return drag_coefficient_function(mach_number)




class rocket_trajectory_optimiser:
    def __init__(self,
                 mission_requirements : dict,
                 physical_constants : dict,
                 design_parameters : dict,
                 mission_profile : dict,
                 delta_v_losses_estimate : float = 2000,
                 plot_bool : bool = False,
                 save_file_path = '/home/jonathanvanzyl/Documents/GitHub/RocketTrajectoryOptimisation/results'):
        self.plot_bool = plot_bool
        self.save_file_path = save_file_path
        
        # Unpack physical constants
        self.mu = physical_constants['mu']                                                      # Gravitational parameter [m^3/s^2]
        self.R_earth = physical_constants['R_earth']                                            # Earth radius [m]
        self.w_earth = physical_constants['w_earth']                                            # Earth angular velocity [rad/s]
        self.g0 = physical_constants['g0']                                                      # Gravity constant on Earth [m/s^2]
        self.scale_height_endo = physical_constants['scale_height_endo']                        # Scale height for endo-atmospheric model [m]
        self.rho0 = physical_constants['rho0']                                                  # Sea level density [kg/m^3]
        self.M_earth = physical_constants['M_earth']                                            # Earth mass [kg]
        self.G = physical_constants['G']                                                        # Gravitational constant [m^3/kg/s^2]

        # Unpack mission requirements
        self.payload_mass = mission_requirements['payload_mass']                                # Payload mass [kg]
        self.mass_fairing = mission_requirements['mass_fairing']                                # Fairing mass [kg]
        self.altitude_orbit = mission_requirements['altitude_orbit']                            # Orbit altitude [m]
        self.semi_major_axis = self.altitude_orbit + self.R_earth                               # Semi-major axis [m]
        self.accel_max_first_stage = mission_requirements['max_first_stage_g'] * self.g0        # Maximum first stage acceleration [m/s^2]
        self.accel_max_second_stage = mission_requirements['max_second_stage_g'] * self.g0      # Maximum second stage acceleration [m/s^2]
        self.max_accelerations = [self.accel_max_first_stage, self.accel_max_second_stage]      # Maximum acceleration for each stage [m/s^2]
        self.number_of_stages = mission_requirements['number_of_stages']                        # Number of stages [int]
        self.structural_coefficients = mission_requirements['structural_coefficients']          # Structural Coefficient [float]
        self.specific_impulses_vacuum = mission_requirements['specific_impulses_vacuum']        # Vacuum Specific Impulse [s]

        # Design parameters
        self.aerodynamic_area = design_parameters['aerodynamic_area']                            # Reference aerodynamic area [m^2] {endo-phase ascent}
        self.nozzle_exit_area = design_parameters['nozzle_exit_area']                            # Exhaust nozzle area [m^2]
        self.nozzle_exit_pressure = design_parameters['nozzle_exit_pressure']                    # Nozzle exit pressure [Pa]

        # Initial conditions
        self.latitude = mission_requirements['launch_site_latitude']                             # Kourou latitude [rad] - launch altitude
        self.position_vector = np.array([self.R_earth * np.cos(self.latitude), 0, self.R_earth * np.sin(self.latitude)])       # Initial position vector [m]
        self.unit_position_vector = self.position_vector / np.linalg.norm(self.position_vector)  # Initial position unit vector
        self.east_vector = np.cross([0, 0, 1], self.unit_position_vector)                        # East vector [m]
        self.unit_east_vector = self.east_vector / np.linalg.norm(self.east_vector)              # East unit vector
        self.velocity_vector = np.cross(self.w_earth, self.position_vector)                      # Initial velocity vector [m/s]
        self.unit_velocity_vector = self.velocity_vector / np.linalg.norm(self.velocity_vector)  # Initial velocity unit vector

        # Mission profile, can optimise these parameters later
        self.target_altitude_vertical_rising = mission_profile['target_altitude_vertical_rising']  # target altitude [m], vertical rising
        self.kick_angle = mission_profile['kick_angle']                                            # kick angle [deg], gravity turn
        self.target_altitude_gravity_turn = mission_profile['target_altitude_gravity_turn']        # target altitude [m], maximum altitude for gravity turn
        self.coasting_time = mission_profile['coasting_time']                                      # coasting time [s], endo-atmospheric
        self.exo_atmoshere_target_altitude_propelled = mission_profile['exo_atmoshere_target_altitude_propelled']  # target altitude [m], exo-atmospheric
        self.minimum_delta_v_adjustments_exo = mission_profile['minimum_delta_v_adjustments_exo']  # minimum delta v adjustments [m/s], exo-atmospheric circularisation

        # Initial state
        self.state = self.rocket_sizing_expendable(delta_v_losses_estimate)
        self.time = 0.0
        self.total_losses = 0.0

        # Results
        self.states = []
        self.times = []
        self.trajectory_results_dict = {}

        # Process trajectory
        self.process_trajectory()

    def reset(self):
        self.state = self.rocket_sizing_expendable()
        self.time = 0.0
        self.total_losses = 0.0
        self.states = []
        self.times = []
        self.trajectory_results_dict = {}
        self.process_trajectory()

    
    def process_trajectory(self):
        print(self.stage_properties_dict)
        self.vertical_rising()
        self.gravity_turn()
        self.endo_coasting()


    def rocket_sizing_expendable(self,
                                 delta_v_losses_estimate = 2000):
        
        initial_mass, self.sub_stage_masses, self.stage_masses, self.structural_masses, \
            self.propellant_masses, delta_v_required, delta_v_required_stages, payload_ratios, mass_ratios = staging_expendable(self.number_of_stages,
                                                                                                                           self.specific_impulses_vacuum,
                                                                                                                           self.structural_coefficients,
                                                                                                                           self.payload_mass,
                                                                                                                           self.semi_major_axis,
                                                                                                                           delta_v_loss = delta_v_losses_estimate,
                                                                                                                           mu = self.mu,
                                                                                                                           g0 = self.g0)
        initial_state = [self.position_vector[0],
                            self.position_vector[1],
                            self.position_vector[2],
                            self.velocity_vector[0],
                            self.velocity_vector[1],
                            self.velocity_vector[2],
                            initial_mass]
        
        stage_properties_dict = compute_stage_properties(initial_mass,
                                                         self.stage_masses,
                                                         self.propellant_masses,
                                                         self.structural_masses,
                                                         self.specific_impulses_vacuum,
                                                         self.g0,
                                                         self.max_accelerations,
                                                         self.mass_fairing)
        
        stage_properties_dict['delta_v_required'] = delta_v_required
        stage_properties_dict['delta_v_required_stages'] = delta_v_required_stages
        stage_properties_dict['payload_ratios'] = payload_ratios
        stage_properties_dict['mass_ratios'] = mass_ratios
        
        self.stage_properties_dict = stage_properties_dict

        return initial_state

    def losses_over_states(self,
                          states,
                          times,
                          endo_atmosphere_bool = True,
                          coasting_bool = False):
        def losses_calculator(t,
                            state_vector,
                            dt,
                            endo_atmosphere_bool,
                            coasting_bool):
            '''
            state_vector: [x, y, z, vx, vy, vz, m]

            reference frame : Inertial Equatorial reference system:
            - X-axis : through meridian passing through launch site.
            - Y-axis : as consequence.
            - Z-axis : through North pole.

            Losses calculator for the rocket trajectory optimisation problem.
            - Gravity losses : \int_{0}^{t} g * sin(gamma) * dt
            - Drag losses : \int_{0}^{t} D/m * dt
            - Pressure losses : \int_{0}^{t} p_a * A_e / m * dt
            - Steering losses : ...
            '''
            # Unpack state vector
            x, y, z, vx, vy, vz, m = state_vector
            position = state_vector[:3]                     # position vector [m]
            vel = state_vector[3:6]                         # velocity vector [m/s]

            gamma = np.arctan2(np.sqrt(vx**2 + vy**2), vz)  # flight path angle [rad]
            altitude = np.linalg.norm([x, y, z]) - self.R_earth  # altitude [m]
            g = self.G * (self.R_earth / (self.R_earth + altitude))**2     # gravity acceleration [m/s^2]
            Lg = g * np.sin(gamma) * dt                     # gravity losses [m/s^2]
            
            # Steering losses
            # Unsure
            Ls = 0

            if endo_atmosphere_bool:
                # ENDO ONLY
                air_density, atmospheric_pressure, speed_of_sound = self.endo_atmospheric_model(altitude) # air density, atmospheric pressure, speed of sound
                vel_rel = vel - np.cross(self.w_earth, position)         # relative velocity vector [m/s]
                mach = np.linalg.norm(vel_rel) / speed_of_sound     # Mach number [-]
                cd = get_drag_coefficient(mach)                     # drag coefficient [-]
                drag = 0.5 * air_density * (np.linalg.norm(vel_rel)**2) * self.aerodynamic_area * cd # drag force [N]
                Ld = drag / m * dt                                  # drag losses [m/s^2]

                # ENDO ONLY
                if not coasting_bool:
                    Lp = atmospheric_pressure * self.nozzle_exit_area / m * dt # pressure losses [m/s^2]
                    losses = Lg + Ld + Lp + Ls
                else:
                    losses = Lg + Ld + Ls

            else:
                # EXO ONLY
                losses = Lg + Ls
            return losses
        
        dt_array = np.diff(times)
        # vertical_rising_states: (7, 1004)
        # vertical_rising_time: (1004,)
        # with 7 states and 1004 time steps
        total_losses = 0
        for i in range(len(times) - 1):
            total_losses += losses_calculator(times[i], states[:, i], dt_array[i], endo_atmosphere_bool, coasting_bool)
        return total_losses

    def endo_atmospheric_model(self, altitude):            # Altitute in meters [m]
        if altitude >= 0:
            rho = self.rho0 * np.exp(-altitude / self.scale_height_endo)
            P_a = 101325 * (rho / self.rho0)
        else:
            rho = self.rho0
            P_a = 101325
        a = 340.29          # Adjust!!!
        return rho, P_a, a

    def rocket_dynamics_endo(self, t, state_vector, mass_flow_endo):
        pos = state_vector[:3]
        vel = state_vector[3:6]
        m = state_vector[6]
        alt = np.linalg.norm(pos) - self.R_earth
        rho, p_atm, a = self.endo_atmospheric_model(alt)
        vel_rel = vel - np.cross(self.w_earth, pos)
        mach = np.linalg.norm(vel_rel) / a
        cd = get_drag_coefficient(mach)
        thrust = self.specific_impulses_vacuum[0] * self.g0 * mass_flow_endo + \
                (self.nozzle_exit_pressure - p_atm) * self.nozzle_exit_area
        drag = 0.5 * rho * (np.linalg.norm(vel_rel)**2) * self.aerodynamic_area * cd
        r_dot = vel
        v_dot = (-self.mu / (np.linalg.norm(pos)**3)) * pos \
                + (thrust / m) * (pos / np.linalg.norm(pos)) \
                - (drag / m) * (pos / np.linalg.norm(pos))
        dm = -mass_flow_endo
        return np.concatenate((r_dot, v_dot, [dm]))
    

    def make_events(self,
                    target_altitude,
                    minimum_mass):
        # Make altitude and fuel mass events
        def make_altitude_event():
            def altitude_event(t, y):
                altitude = np.linalg.norm(y[:3]) - self.R_earth
                return altitude - target_altitude
            altitude_event.terminal = True
            return altitude_event

        def make_mass_flow_event(minimum_mass):
            def mass_flow_event(t, y):
                return y[6] - minimum_mass
            mass_flow_event.terminal = True
            return mass_flow_event
        return [make_altitude_event(), make_mass_flow_event(minimum_mass)]

    def vertical_rising(self):
        minimum_mass = self.stage_properties_dict['burn_out_masses'][0]
        mass_flow_endo = self.stage_properties_dict['mass_flow_rates'][0]
                                    
        # Mock t_span to cover all events
        t_span = [0, 10000]
        # solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
        sol = solve_ivp(
            lambda t, y: self.rocket_dynamics_endo(t, y, mass_flow_endo),
            t_span,  
            self.state,
            events=self.make_events(self.target_altitude_vertical_rising, minimum_mass), 
            max_step=0.1,  # limiting step size for demonstration
            rtol=1e-8,
            atol=1e-8
        )

        # Plot atitude, velocity, and fuel mass next to each other

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(sol.t, np.linalg.norm(sol.y[:3], axis=0) - self.R_earth)
        axs[0].set_ylabel('Altitude [m]')
        axs[0].set_xlabel('Time [s]')
        axs[1].plot(sol.t, np.linalg.norm(sol.y[3:6], axis=0))
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].set_xlabel('Time [s]')
        axs[2].plot(sol.t, sol.y[6])
        axs[2].set_ylabel('Mass [kg]')
        axs[2].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(self.save_file_path + '/endo_atmospheric_vertical_rising.png')
        if self.plot_bool:
            plt.show()
        else:
            plt.close()

        vertical_rising_times = sol.t
        vertical_rising_states = sol.y
        self.state = sol.y[:, -1]
        self.time = sol.t[-1]

        self.states.append(vertical_rising_states)
        self.times.append(vertical_rising_times)

        vertical_rising_losses = self.losses_over_states(vertical_rising_states,
                                            vertical_rising_times,
                                            endo_atmosphere_bool=True)
        
        self.trajectory_results_dict['vertical_rising'] = {'times': vertical_rising_times,
                                                           'states': vertical_rising_states,
                                                           'losses': vertical_rising_losses}
        
        self.total_losses += vertical_rising_losses        
        

    def gravity_turn(self):
        position_vector = self.state[0:3]
        velocity_vector = self.state[3:6]
        mass = self.state[6]

        velocity_vector = velocity_vector + np.linalg.norm(velocity_vector - np.cross(self.w_earth, position_vector)) * np.sin(self.kick_angle) * self.unit_east_vector
        self.state = np.concatenate((position_vector, velocity_vector, [mass]))                      # Initial state for ground tracking frame

        t_span = [self.time, 10000]
        sol = solve_ivp(
            lambda t, y: self.rocket_dynamics_endo(t, y, self.stage_properties_dict['mass_flow_rates'][0]),
            t_span,
            self.state,
            events=self.make_events(self.target_altitude_gravity_turn, self.stage_properties_dict['burn_out_masses'][0]),
            max_step=0.1,  # limiting step size for demonstration
            rtol=1e-8,
            atol=1e-8)
        
        # Plot atitude, velocity, and fuel mass next to each other
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(sol.t, (np.linalg.norm(sol.y[:3], axis=0) - self.R_earth)/1000)
        axs[0].set_ylabel('Altitude [km]')
        axs[0].set_xlabel('Time [s]')
        axs[1].plot(sol.t, (np.linalg.norm(sol.y[3:6], axis=0))/1000)
        axs[1].set_ylabel('Velocity [km/s]')
        axs[1].set_xlabel('Time [s]')
        axs[2].plot(sol.t, sol.y[6])
        axs[2].set_ylabel('Mass [kg]')
        axs[2].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(self.save_file_path + '/endo_atmospheric_gravity_turn.png')
        if self.plot_bool:
            plt.show()
        else:
            plt.close()

        gravity_turn_times = sol.t
        gravity_turn_states = sol.y
        self.state = sol.y[:, -1]
        self.time = sol.t[-1]
        
        self.states.append(gravity_turn_states)
        self.times.append(gravity_turn_times)
        self.time = gravity_turn_times[-1]

        gravity_turn_losses = self.losses_over_states(gravity_turn_states,
                                                    gravity_turn_times,
                                                    endo_atmosphere_bool=True)
        
        self.trajectory_results_dict['gravity_turn'] = {'times': gravity_turn_times,
                                                        'states': gravity_turn_states,
                                                        'losses': gravity_turn_losses}
        
        self.total_losses += gravity_turn_losses

    def coasting_derivatives(self,
                             t, y):
        r = y[:3]
        v = y[3:6]
        m = y[6]
        rdot = v
        vdot = -self.mu / (np.linalg.norm(r) ** 3) * r
        return np.concatenate((rdot, vdot, [0]))

    def endo_coasting(self):
        position_vector = self.state[0:3]
        velocity_vector = self.state[3:6]
        mass = self.sub_stage_masses[1] - self.mass_fairing # Mass of the second stage & payload - fairing [kg]

        self.state = np.concatenate((position_vector, velocity_vector, [mass]))                      # Initial state for ground tracking frame

        t_span = [self.time, self.coasting_time + self.time]
        sol = solve_ivp(
            self.coasting_derivatives,
            t_span,
            self.state,
            max_step=0.1,
            rtol=1e-8,
            atol=1e-8
        )
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].plot(sol.t, (np.linalg.norm(sol.y[:3], axis=0) - self.R_earth)/1000)
        axs[0].set_ylabel('Altitude [km]')
        axs[0].set_xlabel('Time [s]')
        axs[1].plot(sol.t, (np.linalg.norm(sol.y[3:6], axis=0))/1000)
        axs[1].set_ylabel('Velocity [km/s]')
        axs[1].set_xlabel('Time [s]')
        axs[2].plot(sol.t, sol.y[6])
        axs[2].set_ylabel('Mass [kg]')
        axs[2].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(self.save_file_path + '/endo_atmosphere_coasting.png')
        if self.plot_bool:
            plt.show()
        else:
            plt.close()

        endo_coasting_times = sol.t
        endo_coasting_states = sol.y
        self.state = sol.y[:, -1]
        self.time = sol.t[-1]
        
        self.states.append(endo_coasting_states)
        self.times.append(endo_coasting_times)
        self.time = endo_coasting_times[-1]

        endo_coasting_losses = self.losses_over_states(endo_coasting_states,
                                                  endo_coasting_times,
                                                  endo_atmosphere_bool=True,
                                                  coasting_bool=True)
        
        self.trajectory_results_dict['endo_coasting'] = {'times': endo_coasting_times,
                                                        'states': endo_coasting_states,
                                                        'losses': endo_coasting_losses}
        
        self.total_losses += endo_coasting_losses

if __name__ == '__main__':
    rocket_trajectory_optimiser = rocket_trajectory_optimiser(mission_requirements,
                                                             physical_constants,
                                                             design_parameters,
                                                             mission_profile,
                                                             plot_bool = True)