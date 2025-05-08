import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


degree = 4 # 3 works 4 is better
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
        print(f"Initial velocity {abs(v_y_0):.2f} m/s exceeds limit {v_limit_fn(y_0):.2f} m/s at y_0")
        print("Adjusting to use 99% of limit at y_0")
        target_v_at_y0 = v_limit_fn(y_0) * 0.99  # Use 99% of limit to ensure feasibility
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
    
    # Suppress SciPy optimizer warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Try multiple optimization methods in case one fails
    methods = ['SLSQP', 'trust-constr']
    result = None
    success = False
    
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
            result = minimize(
                objective,
                initial_guess,
                constraints=constraints,
                method=method,
                options={'disp': False}  # Turn off verbose output
            )
            
            if result.success:
                success = True
                print(f"Optimization successful using {method}")
                break
                
        except Exception as e:
            pass
    
    # If all methods failed, try a fallback approach
    if not success and result is not None:
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
                options={'disp': False}
            )
            
            if result.success:
                success = True
                print("Optimization successful using fallback strategy")
        except Exception:
            pass
    
    if not success or result is None:
        print("Using simple linear velocity profile as fallback")
        fallback_slope = target_v_at_y0 / y_0
        return lambda y: fallback_slope * y, y_0 / 2
    
    a_opt, b_opt = result.x
    print(f"Optimal velocity profile: v(y) = {a_opt:.6e}*y^2 + {b_opt:.6e}*y")
    
    # Create optimal velocity function
    v_opt = lambda y: a_opt * y**2 + b_opt * y
    
    # Calculate approximate tangency point
    diffs = v_limit_fn(y_samples) - v_opt(y_samples)
    tangent_idx = np.argmin(np.abs(diffs))
    y_tangent = y_samples[tangent_idx]
    
    # Calculate time of flight (numerical integration)
    dy = 0.1  # small step for numerical integration
    y_steps = np.arange(0.1, y_0, dy)  # Start slightly above zero to avoid division by zero
    times = dy / v_opt(y_steps)  # dt = dy/v
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


# From vy_opt plot and y_fit plot, find the desired acceleration profile
# First, calculate dv/dy (gradient of velocity with respect to altitude)
dv_dy = np.gradient(v_opt_plot, y_fit)

# To get true acceleration (dv/dt), use chain rule: dv/dt = dv/dy * dy/dt = dv/dy * v
# Since v is velocity and dy/dt = v
a_opt_fn = dv_dy * v_opt_plot
a_opt_gs = a_opt_fn / g_0

# Now a_opt_fn of the max_v_s is the max_a_s
dv_dy_max_v_s = np.gradient(poly(y_fit), y_fit)
a_max_v_s = dv_dy_max_v_s * poly(y_fit)
a_max_v_s_gs = a_max_v_s / g_0

# Plot
plt.figure(figsize=(20, 10))
plt.suptitle('Optimal Feasible Landing Trajectory', fontsize=22)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
ax1 = plt.subplot(gs[0])
ax1.plot(y_refs/1000, max_v_s/1000, color='blue', linewidth=4, label='Max Velocity Limit')
ax1.plot(y_fit/1000, poly(y_fit)/1000, '--', color='grey', label='Polyfit (Limit)')
ax1.plot(y_fit/1000, v_opt_plot/1000, 'r-', linewidth=3, label='Optimal Trajectory')
ax1.scatter(y_0/1000, abs(v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
ax1.scatter(y_0/1000, abs(v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
ax1.plot(y_refs/1000, vy_no_thrust/1000, color='black', linewidth=2, linestyle='--', label='No Thrust Velocity')
ax1.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
ax1.set_ylabel(r'v [$km/s$]', fontsize=20)
ax1.set_ylim(0, 2)
ax1.set_title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory', fontsize=20)
ax1.grid(True)
ax1.legend(fontsize=20)
ax1.tick_params(labelsize=16)

ax2 = plt.subplot(gs[1])
ax2.plot(y_fit/1000, a_opt_gs, 'r-', linewidth=3, label='True Acceleration')
ax2.plot(y_fit/1000, a_max_v_s_gs, 'b--', linewidth=3, label='Max Acceleration')
ax2.set_xlabel(r'y [$km$]', fontsize=20)
ax2.set_ylabel(r'a [$g_0$]', fontsize=20)
ax2.set_title('True Acceleration (dv/dt) vs. Altitude', fontsize=20)
ax2.grid(True)
ax2.legend(fontsize=20)
ax2.tick_params(labelsize=16)
ax2.set_ylim(0, 5)

plt.savefig('optimal_trajectory.png')
plt.show()


# a = T/(m_0 - mdot * s)

# a(t) = T/m(t) * tau(t) - g_0 + 0.5 * rho(y(t)) * v(t)^2 * C_n_0 * S
# m(t) = m_0 - mdot * int_0^t tau(t) dt
# v(t) = v_0 + int_0^t a(t) dt
# y(t) = y_0 + int_0^t v(t) dt
# Constraints : m > ms, v(t) < v_max(y(t))
# Initial conditions : m(0) = m_0, v(0) = v_0, y(0) = y_0
# Final conditions : v(t_f) = 0, y(t_f) = 0
# tau(t) = (0,1)

# Minimize int_0^t1 of - |v(t)| * dt i.e. maximise the area under the velocity curve

