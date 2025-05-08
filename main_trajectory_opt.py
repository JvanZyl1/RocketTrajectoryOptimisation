import math
import numpy as np
import matplotlib.pyplot as plt
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.base_environment import load_landing_burn_initial_state

g_0 = 9.80665 # [m/s^2]
dt = 0.01 # [s]

def easy_dynamics(y_0, vy_0, y_1):
    vy_1 = math.sqrt(vy_0**2 + 2*g_0*(y_0 - y_1))
    return vy_1

# x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant,time
state_initial = load_landing_burn_initial_state()
y_0 = state_initial[1]
v_x_0 = state_initial[2]
v_y_0 = state_initial[3]
v_0 = np.sqrt(v_x_0**2 + v_y_0**2)

y_refs = np.linspace(y_0, 0, 100)
air_densities = np.zeros(len(y_refs))
max_v_s = np.zeros(len(y_refs))
max_q = 30000 # [Pa]
no_thrust_velocities = np.zeros(len(y_refs))
vy_no_thrust = np.zeros(len(y_refs))
for i, y_ref in enumerate(y_refs):
    air_densities[i], atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_ref)
    max_v_s[i] = np.sqrt(2* max_q /air_densities[i])
    vy_no_thrust[i] = easy_dynamics(y_0, v_y_0, y_ref)


degree = 3 # 3 works 4 is better
coeffs = np.polyfit(y_refs, max_v_s, degree)
poly = np.poly1d(coeffs)
print(f'poly {poly}')

# Plotting
y_fit = np.linspace(min(y_refs), max(y_refs), 500)
v_fit = poly(y_fit)

from scipy.optimize import minimize

# minimise : - int_t0^t1 of - |v(t)| * dt i.e. maximise the integral of v(t)
# constrained by v(t_0) = v_y_0, y(t_0) = y_0, y(t_1) = 0, v(t_1) = 0
# And must always be below the poly fit
# Likely a second order polynomial

def compute_optimal_trajectory():
    # Define a second-order polynomial for velocity as a function of altitude
    # v(y) = a*y^2 + b*y + c
    # Initial conditions: v(0) = 0, v(y_0) = abs(v_y_0)
    # Using v(0) = 0 => c = 0, so v(y) = a*y^2 + b*y
    
    # Create a parametric velocity function
    def v_profile(y, params):
        a, b = params
        return a * y**2 + b * y
    
    # Velocity limit function using polynomial fit
    v_limit_fn = lambda y: poly(y)
    
    # Check if abs(v_y_0) exceeds the velocity limit at y_0
    if abs(v_y_0) > v_limit_fn(y_0):
        print(f"Warning: Initial velocity {abs(v_y_0):.2f} m/s exceeds limit {v_limit_fn(y_0):.2f} m/s at y_0")
        print("Using velocity limit at y_0 as the constraint")
        target_v_at_y0 = min(abs(v_y_0), v_limit_fn(y_0) * 0.99)  # Use 99% of limit to ensure feasibility
    else:
        target_v_at_y0 = abs(v_y_0)
    
    # Sample points for constraint checking
    y_samples = np.linspace(0, y_0, 200)
    
    # Objective function: minimize -∫v(y)dy (equivalent to minimizing time)
    # For v(y) = a*y^2 + b*y, the integral is (a/3)*y^3 + (b/2)*y^2
    def objective(params):
        a, b = params
        # The negative sign is because we're minimizing, and we want to maximize
        # the area under the velocity curve (to minimize time)
        return -((a/3) * y_0**3 + (b/2) * y_0**2)
    
    # Constraint 1: v(y_0) = target_v_at_y0 (could be abs(v_y_0) or lower to ensure feasibility)
    def constraint_initial_velocity(params):
        a, b = params
        return v_profile(y_0, params) - target_v_at_y0
    
    # Constraint 2: v(y) ≤ v_limit(y) for all y
    def constraint_velocity_limit(params):
        a, b = params
        # Return an array of velocity differences that must all be non-negative
        # to satisfy v(y) ≤ v_limit(y)
        return v_limit_fn(y_samples) - v_profile(y_samples, params)
    
    # Try multiple optimization methods in case one fails
    methods = ['SLSQP', 'trust-constr']
    
    for method in methods:
        try:
            # Initial guess: linear profile from (0,0) to (y_0, target_v_at_y0)
            initial_guess = [0, target_v_at_y0 / y_0]
            
            # Set up constraints
            constraints = [
                {'type': 'eq', 'fun': constraint_initial_velocity},
                {'type': 'ineq', 'fun': constraint_velocity_limit}
            ]
            
            # Run optimization
            print(f"Trying optimization with method: {method}")
            result = minimize(
                objective,
                initial_guess,
                constraints=constraints,
                method=method,
                options={'disp': True}
            )
            
            if result.success:
                break
            else:
                print(f"Method {method} failed: {result.message}")
                
        except Exception as e:
            print(f"Error with method {method}: {str(e)}")
    
    # If all methods failed, try a fallback approach without initial velocity constraint
    if not result.success:
        print("All optimization methods failed. Trying fallback approach...")
        
        # Fallback: Try optimization without the equality constraint, just stay under the limit
        try:
            # New initial guess (conservative)
            conservative_guess = [0, 0.5 * target_v_at_y0 / y_0]
            
            # Only use the velocity limit constraint
            constraints = [
                {'type': 'ineq', 'fun': constraint_velocity_limit}
            ]
            
            result = minimize(
                objective,
                conservative_guess,
                constraints=constraints,
                method='SLSQP',
                options={'disp': True}
            )
            
            print("Fallback results:", "Success" if result.success else "Failed")
        except Exception as e:
            print(f"Fallback approach failed: {str(e)}")
    
    if not result.success:
        print("All optimization approaches failed.")
        # Return a linear fallback profile as a last resort
        fallback_slope = min(target_v_at_y0 / y_0, 
                           min(v_limit_fn(y_samples) / y_samples[1:]))  # avoid division by zero
        return lambda y: fallback_slope * y, y_0 / 2
    
    a_opt, b_opt = result.x
    print(f"Optimal parameters: a={a_opt:.6e}, b={b_opt:.6e}")
    
    # Create optimal velocity function
    v_opt = lambda y: a_opt * y**2 + b_opt * y
    
    # Calculate approximate tangency point
    diffs = v_limit_fn(y_samples) - v_opt(y_samples)
    tangent_idx = np.argmin(np.abs(diffs))
    y_tangent = y_samples[tangent_idx]
    
    # Calculate time of flight (numerical integration)
    dy = 0.1  # small step for numerical integration
    y_steps = np.arange(0, y_0, dy)
    times = dy / v_opt(y_steps[1:])  # dt = dy/v, skip y=0 to avoid division by zero
    total_time = np.sum(times)
    
    print(f"Estimated time of flight: {total_time:.2f} seconds")
    print(f"Tangency point at altitude: {y_tangent/1000:.2f} km")
    
    return v_opt, y_tangent

# Compute optimal trajectory
try:
    v_opt_result = compute_optimal_trajectory()
    if v_opt_result:
        v_opt_fn, y_tangent = v_opt_result
        # Generate points for plotting
        v_opt_plot = v_opt_fn(y_fit)
    else:
        # Fallback to linear profile if optimization completely failed
        v_opt_plot = (abs(v_y_0) / y_0) * y_fit
except Exception as e:
    print(f"Error in optimization: {str(e)}")
    # Fallback to linear profile
    v_opt_plot = (abs(v_y_0) / y_0) * y_fit

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_refs/1000, max_v_s/1000, color='blue', linewidth=4, label='Max Velocity Limit')
plt.plot(y_fit/1000, poly(y_fit)/1000, '--', color='grey', label='Polyfit (Limit)')
plt.plot(y_fit/1000, v_opt_plot/1000, 'r-', linewidth=3, label='Optimal Trajectory')
plt.scatter(y_0/1000, abs(v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
plt.scatter(y_0/1000, abs(v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
plt.plot(y_refs/1000, vy_no_thrust/1000, color='black', linewidth=2, linestyle='--', label='No Thrust Velocity')
plt.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
plt.xlabel('y [km]')
plt.ylabel('v [km/s]')
plt.ylim(0, 2)
plt.title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory')
plt.grid(True)
plt.legend()
plt.show()
