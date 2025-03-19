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
# Using conditions: a0 + a1*T + a2*T**2 + a3*T**3 = 45 and a1 + 2*a2*T + 3*a3*T**2 = 0.
# Solve for a2 and a3.
a3 = 90 / (T**3)
a2 = -1.5 * a3 * T
theta_ref = a0 + a1 * t + a2 * t**2 + a3 * t**3

# Parameters
d_tcg =  60

T_e = 2745*1000 # Engine thrust [N]
n_eg = 3 + 12   # Number of engines gimballed
p_e = 100000    # Nozzle exit pressure [Pa]
p_a = 101325    # Ambient pressure [Pa]
A_e = 0.01      # Engine nozzle exit area [m^2]

T_e_with_losses = T_e + (p_e - p_a) * A_e

T_g = T_e_with_losses * n_eg
I_z = 2e10

# Simulation of cascaded PID controller
def simulate(pid_params):
    Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate = pid_params

    theta = 90.0
    theta_dot = 0.0
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
        theta_g = np.arcsin(u/2)
        
        theta_dot_dot = (d_tcg * T_g / I_z) * u
        theta_dot += theta_dot_dot * dt
        theta += theta_dot * dt
        
        prev_error_rate = error_rate
        theta_hist[i] = theta

    # Cost: sum of squared tracking errors
    cost = np.sum((theta_ref - theta_hist)**2)
    print(f'Cost: {cost}')
    return cost

# Initial guess for gains: [Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate]
pid0 = [0.05, 0.001, 0.8, 0.02, 0.05]

res = minimize(simulate, pid0, bounds=[(0, None)]*5, options={'maxiter':200})
opt_pid = res.x
print("Optimized PID gains:")
print("Kp_theta = {:.4f}, Ki_theta = {:.4f}, Kp_rate = {:.4f}, Ki_rate = {:.4f}, Kd_rate = {:.4f}".format(*opt_pid))

# Run simulation with optimized gains
Kp_theta, Ki_theta, Kp_rate, Ki_rate, Kd_rate = opt_pid
theta = 90.0
theta_dot = 0.0
error_theta_int = 0.0
error_rate_int = 0.0
prev_error_rate = 0.0

theta_hist = np.zeros_like(t)
theta_dot_hist = np.zeros_like(t)
u_hist = np.zeros_like(t)

for i in range(len(t)):
    error_theta = theta_ref[i] - theta
    error_theta_int += error_theta * dt
    theta_rate_cmd = Kp_theta * error_theta + Ki_theta * error_theta_int

    error_rate = theta_rate_cmd - theta_dot
    error_rate_int += error_rate * dt
    d_error_rate = (error_rate - prev_error_rate) / dt
    u = Kp_rate * error_rate + Ki_rate * error_rate_int + Kd_rate * d_error_rate
    u = np.clip(u, -1, 1)
    theta_g = np.arcsin(u/2)
    
    theta_dot_dot = (1 * 1 / 1) * u
    theta_dot += theta_dot_dot * dt
    theta += theta_dot * dt
    
    prev_error_rate = error_rate
    theta_hist[i] = theta
    theta_dot_hist[i] = theta_dot
    u_hist[i] = u

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, theta_hist, 'b', label='Actual')
plt.plot(t, theta_ref, 'r--', label='Reference')
plt.ylabel('Pitch (deg)')
plt.title('Attitude Tracking')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, theta_dot_hist, 'b')
plt.ylabel('Pitch Rate (deg/s)')
plt.title('Pitch Rate')

plt.subplot(3,1,3)
plt.plot(t, u_hist, 'b')
plt.ylabel('Control Input u')
plt.xlabel('Time (s)')
plt.title('Control Input')
plt.tight_layout()
plt.show()
