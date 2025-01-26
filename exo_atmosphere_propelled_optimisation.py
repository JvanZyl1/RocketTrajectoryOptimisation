import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from params import mu

def exo_dynamics(t,
           state_vector, #[x, y, z, vx, vy, vz, m, pr_x, pr_y, pr_z, pv_x, pv_y, pv_z]
           second_stage_mass_flow,
           second_stage_thrust):
    # Extract state variables
    r = state_vector[0:3]       # Position Vector, meters
    v = state_vector[3:6]       # Velocity Vector, meters/second
    m = state_vector[6]         # Mass, kilograms
    pr = state_vector[7:10]     # Position adjoint multipliers
    pv = state_vector[10:13]    # Velocity adjoint multipliers

    # Find unit vectors
    pv_unit = pv / np.linalg.norm(pv)
    r_unit = r / np.linalg.norm(r)

    v_dot = - (mu / np.linalg.norm(r)**2) * r_unit + (second_stage_thrust / m) * pv_unit
    pr_dot = - (mu / np.linalg.norm(r)**3) * (3 * np.dot(pv, r) * r / np.linalg.norm(r)**2 - pv)
    pv_dot = - pr
    m_dot = -second_stage_mass_flow

    return np.concatenate((v, v_dot, [m_dot], pr_dot, pv_dot))

def objective(x_optimisation,
              initial_state,
              second_stage_thrust,
              second_stage_mass_flow,
              desired_r):
    # x_optimisation = [scaled_prop_time, prU, pvU]
    prop_time = x_optimisation[0]
    prU = x_optimisation[1:4]
    pvU = x_optimisation[4:7]
    
    # Normalize the unit vectors
    if np.linalg.norm(pvU) == 0:
        pvU = np.array([0.0, 0.0, 0.0])
    else:
        pvU = pvU / np.linalg.norm(pvU)
    
    if np.linalg.norm(prU) == 0:
        prU = np.array([0.0, 0.0, 0.0])
    else:
        prU = prU / np.linalg.norm(prU)
    
    # Initial augmented state
    augmented_initial_state = np.concatenate((initial_state, prU, pvU))
    
    # Simulate dynamics over propulsion time
    t_span = (0, prop_time)
    t_eval = [prop_time]  # Only evaluate at final time
    
    sol = solve_ivp(
        fun=lambda t, y: exo_dynamics(t, y, second_stage_mass_flow, second_stage_thrust),
        t_span=t_span,
        y0=augmented_initial_state,
        method='RK45',
        t_eval=t_eval,
        vectorized=False
    )
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return np.inf  # Penalize failed integration
    
    # Extract final state
    final_state = sol.y[:, -1]
    final_r = final_state[0:3]
    final_v = final_state[3:6]
    #final_m = final_state[6]
    
    # Define desired orbit parameters (e.g., circular orbit at rp)
    desired_v = np.sqrt(mu / desired_r)  # Circular orbit velocity
    
    # Calculate the difference from desired orbit
    r_error = np.linalg.norm(final_r) - desired_r
    v_error = np.linalg.norm(final_v) - desired_v
    
    # Define penalty for not achieving desired orbit
    penalty = r_error**2 + v_error**2
    
    # Objective: Minimize propulsion time plus penalties
    return prop_time + penalty

def optimisise_exo_atmospheric_phase(initial_state,
                                     initial_optimisation_variables,
                                     bounds,
                                     desired_r,
                                     second_stage_thrust,
                                     second_stage_mass_flow):
    # Initial state: [r, v, m]
    # Initial optimisation variables: [burn time, prU, pvU]
    # Perform the optimization
    result = minimize(
        fun=lambda x: objective(x, initial_state, second_stage_thrust, second_stage_mass_flow, desired_r),
        x0=initial_optimisation_variables,
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True,
            'maxiter': 1000
        }
    )
    ##### DO THIS OTHER STUFF #####
    if result.success:
        X_opt = result.x
        optimized_prop_time = X_opt[0] * 1e2  # Scale back to seconds
        optimized_prU = X_opt[1:4] / np.linalg.norm(X_opt[1:4]) if np.linalg.norm(X_opt[1:4]) != 0 else X_opt[1:4]
        optimized_pvU = X_opt[4:7] / np.linalg.norm(X_opt[4:7]) if np.linalg.norm(X_opt[4:7]) != 0 else X_opt[4:7]
        
        print("Optimization Successful!")
        print(f"Optimized Propulsion Time: {optimized_prop_time:.2f} seconds")
        print(f"Optimized Propulsion Direction (prU): {optimized_prU}")
        print(f"Optimized Velocity Direction (pvU): {optimized_pvU}")
        
        # Simulate final trajectory with optimized variables
        second_stage_thrust = 1e4  # Newtons (example value)
        second_stage_mass_flow = 10.0  # kg/s (example value)
        
        # Define final state vector with optimized variables
        final_state_aug = np.concatenate((state0, optimized_prU, optimized_pvU))
        
        # Simulate dynamics over propulsion time
        t_span_final = (0, optimized_prop_time)
        t_eval_final = [optimized_prop_time]  # Only evaluate at final time
        
        sol_final = solve_ivp(
            fun=lambda t, y: exo_dynamics(t, y, second_stage_mass_flow, second_stage_thrust),
            t_span=t_span_final,
            y0=final_state_aug,
            method='RK45',
            t_eval=t_eval_final,
            vectorized=False
        )
        
        if sol_final.success:
            final_state = sol_final.y[:, -1]
            final_r = final_state[0:3]
            final_v = final_state[3:6]
            final_m = final_state[6]
            
            print(f"Final Position: {final_r} meters")
            print(f"Final Velocity: {final_v} m/s")
            print(f"Final Mass: {final_m} kg")
            
            # Desired orbit parameters
            desired_r = 8000e3  # meters
            desired_v = np.sqrt(mu / desired_r)  # Circular orbit velocity
            
            # Calculate errors
            r_error = np.linalg.norm(final_r) - desired_r
            v_error = np.linalg.norm(final_v) - desired_v
            
            print(f"Final Orbit Radius Error: {r_error:.2f} meters")
            print(f"Final Orbit Velocity Error: {v_error:.2f} m/s")
        else:
            print(f"Final trajectory simulation failed: {sol_final.message}")
    else:
        print("Optimization Failed.")
        print(f"Reason: {result.message}")

