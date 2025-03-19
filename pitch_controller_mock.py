import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Simulation parameters
dt = 0.01
T_final = 120
t = np.arange(0, T_final + dt, dt)
T = T_final

# Cubic polynomial reference trajectory for pitch:
# Boundary: theta(0)=90, theta_dot(0)=0, theta(T)=45, theta_dot(T)=0.
a0 = 90
a1 = 0
a3 = 90 / (T**3)
a2 = -1.5 * a3 * T
theta_ref = a0 + a1 * t + a2 * t**2 + a3 * t**3

# Parameters for the rocket (example values)
d_tcg = 60

T_e = 2745*1000    # Engine thrust [N]
n_eg = 3 + 12      # Number of gimballed engines
p_e = 100000       # Nozzle exit pressure [Pa]
p_a = 101325       # Ambient pressure [Pa]
A_e = 0.01         # Engine nozzle exit area [m^2]

T_e_with_losses = T_e + (p_e - p_a) * A_e
T_g = T_e_with_losses * n_eg
I_z = 2e10

# Gimbal lag time constant (seconds)
tau = 1.0

# Simulation of cascaded PID controller with gimbal dynamics
def simulate(pid_params):
    Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate = pid_params
    
    theta = 90.0
    theta_dot = 0.0
    theta_g = 0.0  # initialize actual gimbal angle (rad)
    error_theta_int = 0.0
    error_rate_int = 0.0
    prev_error_rate = 0.0

    theta_hist = np.zeros_like(t)
    
    for i in range(len(t)):
        error_theta = theta_ref[i] - theta
        error_theta_int += error_theta * dt
        theta_rate_cmd = Kp_theta * error_theta + Ki_theta * error_theta_int
        
        error_rate = theta_rate_cmd - theta_dot
        error_rate_int += error_rate * dt
        d_error_rate = (error_rate - prev_error_rate) / dt
        u = Kp_rate * error_rate + Ki_rate * error_rate_int + Kd_rate * d_error_rate
        u = np.clip(u, -1, 1)
        # Compute desired gimbal angle command
        theta_g_cmd = np.arcsin(u/2)
        # Gimbal lag dynamics: first order filter with time constant tau
        theta_g += dt * (theta_g_cmd - theta_g) / tau
        
        # Use actual gimbal angle in dynamics (u_eff = 2 sin(theta_g))
        u_eff = 2 * np.sin(theta_g)
        theta_dot_dot = (d_tcg * T_g / I_z) * u_eff
        theta_dot += theta_dot_dot * dt
        theta += theta_dot * dt
        
        prev_error_rate = error_rate
        theta_hist[i] = theta

    cost = np.sum((theta_ref - theta_hist)**2)
    print(f'Cost: {cost}')
    return cost

# Initial guess for gains: [Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate]
pid0 = [0.05, 0.001, 0.8, 0.02, 0.05]

res = minimize(simulate, pid0, bounds=[(0, None)]*5, options={'maxiter':200})
opt_pid = res.x
print("Optimized PID gains:")
print("Kp_theta = {:.4f}, Ki_theta = {:.4f}, Kp_rate = {:.4f}, Ki_rate = {:.4f}, Kd_rate = {:.4f}".format(*opt_pid))

# Run simulation with optimized gains using gimbal dynamics in the inner loop
Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate = opt_pid
theta = 90.0
theta_dot = 0.0
theta_g = 0.0  # initial actual gimbal angle
error_theta_int = 0.0
error_rate_int = 0.0
prev_error_rate = 0.0

theta_hist = np.zeros_like(t)
theta_dot_hist = np.zeros_like(t)
u_hist = np.zeros_like(t)
theta_g_hist = np.zeros_like(t)

for i in range(len(t)):
    error_theta = theta_ref[i] - theta
    error_theta_int += error_theta * dt
    theta_rate_cmd = Kp_theta * error_theta + Ki_theta * error_theta_int

    error_rate = theta_rate_cmd - theta_dot
    error_rate_int += error_rate * dt
    d_error_rate = (error_rate - prev_error_rate) / dt
    u = Kp_rate * error_rate + Ki_rate * error_rate_int + Kd_rate * d_error_rate
    u = np.clip(u, -1, 1)
    
    theta_g_cmd = np.arcsin(u/2)
    theta_g += dt * (theta_g_cmd - theta_g) / tau
    
    # Effective control: u_eff = 2*sin(theta_g)
    u_eff = 2 * np.sin(theta_g)
    theta_dot_dot = (d_tcg * T_g / I_z) * u_eff
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    
    prev_error_rate = error_rate
    theta_hist[i] = theta
    theta_dot_hist[i] = theta_dot
    u_hist[i] = u
    theta_g_hist[i] = theta_g

plt.figure(figsize=(10,10))
plt.subplot(4,1,1)
plt.plot(t, theta_hist, 'b', label='Actual')
plt.plot(t, theta_ref, 'r--', label='Reference')
plt.ylabel('Pitch (deg)')
plt.title('Attitude Tracking')
plt.legend()

plt.subplot(4,1,2)
plt.plot(t, theta_dot_hist, 'b')
plt.ylabel('Pitch Rate (deg/s)')
plt.title('Pitch Rate')

plt.subplot(4,1,3)
plt.plot(t, u_hist, 'b')
plt.ylabel('Control Input u')
plt.title('Control Input (PID output)')

plt.subplot(4,1,4)
plt.plot(t, np.degrees(theta_g_hist), 'b')
plt.ylabel('Gimbal Angle (deg)')
plt.xlabel('Time (s)')
plt.title('Gimbal Dynamics')
plt.tight_layout()
plt.show()
