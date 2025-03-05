import numpy as np

mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
R_earth = 6378137  # Earth radius [m]
w_earth = np.array([0, 0, 2 * np.pi / 86164])  # Earth angular velocity [rad/s]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def cost_fcn(optimisation_state,
             simulate_lambda_func,
             final_mass_compute_lambda_func,
             dry_mass,
             semi_major_axis):
        """
        Objective function for optimization.
        optimisation_state = [prop_time_scaled, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
        """
        # Final state vector at arrival : [r_x, r_y, r_z, v_x, v_y, v_z, m
        final_state, states, times = simulate_lambda_func(optimisation_state)
        altitude = np.linalg.norm(final_state[0:3]) - R_earth

        mass_final_at_circular = final_mass_compute_lambda_func(final_state)

        J = (mass_final_at_circular - dry_mass)/mass_final_at_circular  # >= 0
        #print(f'Cost function: {J}, Altitude: {altitude}')
        return J    