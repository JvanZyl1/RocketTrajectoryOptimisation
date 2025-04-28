import numpy as np
import math
import random
from scipy.signal import cont2discrete

class VonKarmanFilter:
    """Longitudinal or lateral Von Kármán gust: second-order shaping filter."""
    def __init__(self, L: float, sigma: float, V: float, dt: float):
        omega0 = V / L
        zeta = 1 / math.sqrt(2)          # damping ratio
        A_c = np.array([[0, 1],
                        [-omega0**2, -2*zeta*omega0]])
        B_c = np.array([[0],
                        [sigma * math.sqrt(2 * omega0**3 / math.pi)]])
        C_c = np.array([[1, 0]])
        D_c = np.zeros((1, 1))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C_c, D_c), dt)
        self.Ad, self.Bd, self.Cd = A_d, B_d.flatten(), C_d.flatten()
        self.state = np.zeros(2)

    def step(self) -> float:
        w = np.random.randn()
        self.state = self.Ad @ self.state + self.Bd * w
        return self.Cd @ self.state
    
    def reset(self):
        self.state = np.zeros(2)

class VKDisturbanceGenerator:
    """
    Returns additive body-axis force vector [N] and pitching moment [N*m].
    Gust lift/drag scaling assumes small-angle gusts; adapt for high-AoA regimes.
    """
    def __init__(self, dt: float, V: float,
                 frontal_area: float):
        self.L_u_min = 100.0
        self.L_u_max = 500.0
        self.L_v_min = 30.0
        self.L_v_max = 300.0
        self.sigma_u_min = 0.5
        self.sigma_u_max = 2.0
        self.sigma_v_min = 0.5
        self.sigma_v_max = 2.0

        self.frontal_area = frontal_area
        self.V = V
        self.dt = dt
        self.u_filter, self.v_filter = self.create_random_disturbance()

    def create_random_disturbance(self):
        self.L_u = random.uniform(self.L_u_min, self.L_u_max)
        self.L_v = random.uniform(self.L_v_min, self.L_v_max)
        self.sigma_u = random.uniform(self.sigma_u_min, self.sigma_u_max)
        self.sigma_v = random.uniform(self.sigma_v_min, self.sigma_v_max)
        u_filter = VonKarmanFilter(self.L_u, self.sigma_u, self.V, self.dt)
        v_filter = VonKarmanFilter(self.L_v, self.sigma_v, self.V, self.dt)
        return u_filter, v_filter
        

    def __call__(self, state, t, rho, q_dyn, speed, d_cp_cg, **kwargs):
        gust_u = self.u_filter.step()           # body-axis forward gust [m s⁻¹]
        gust_v = self.v_filter.step()           # body-axis normal gust   [m s⁻¹]
        dF = 0.5 * rho * speed * np.array([gust_u, gust_v]) * self.frontal_area
        dM = 0.0                                # thrust-vector error etc. may be added here
        return dF, dM

    def reset(self):
        self.u_filter, self.v_filter = self.create_random_disturbance()