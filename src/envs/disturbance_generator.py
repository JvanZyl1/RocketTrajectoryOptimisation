import numpy as np
import math
import random
from scipy.signal import cont2discrete

class VonKarmanFilter:
    """Second-order shaping filter for Von Kármán gust."""
    def __init__(self, L: float, sigma: float, V: float, dt: float):
        self.L = L
        self.sigma = sigma
        self.V = V
        self.dt = dt
        self.initialise_filter()

    def initialise_filter(self):
        self.omega0 = self.V / self.L
        zeta = 1 / math.sqrt(2)
        scale = math.sqrt(math.pi / (2 * self.omega0**3))
        A_c = np.array([[0, 1],
                        [-self.omega0**2, -2 * zeta * self.omega0]])
        B_c = np.array([[0],
                        [self.sigma * scale]])
        C_c = np.array([[0, 1]])
        D_c = np.zeros((1, 1))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C_c, D_c), self.dt, method='bilinear')
        self.Ad = A_d
        self.Bd = B_d.flatten()
        self.Cd = C_d
        self.state = np.zeros(2)

    def step(self) -> float:
        w = np.random.randn()
        self.state = self.Ad @ self.state + self.Bd * w
        return float(self.Cd @ self.state)

    def reset(self):
        self.initialise_filter()

class VKDisturbanceGenerator:
    """Generates body-axis disturbance force and moment."""
    def __init__(self, dt: float, V: float, frontal_area: float):
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
        self._initial_np_rng_state = np.random.get_state()
        self.u_filter, self.v_filter = self.create_random_disturbance()
        self._initial_u_state = self.u_filter.state.copy()
        self._initial_v_state = self.v_filter.state.copy()
        self.prev_rho = None

    def create_random_disturbance(self):
        if random.random() < 0.5:
            self.L_u = random.uniform(self.L_u_min, (self.L_u_min + self.L_u_max) / 2)
            self.L_v = random.uniform((self.L_v_min + self.L_v_max) / 2, self.L_v_max)
        else:
            self.L_u = random.uniform((self.L_u_min + self.L_u_max) / 2, self.L_u_max)
            self.L_v = random.uniform(self.L_v_min, (self.L_v_min + self.L_v_max) / 2)
        if random.random() < 0.5:
            self.sigma_u = random.uniform(self.sigma_u_min, (self.sigma_u_min + self.sigma_u_max) / 2)
            self.sigma_v = random.uniform((self.sigma_v_min + self.sigma_v_max) / 2, self.sigma_v_max)
        else:
            self.sigma_u = random.uniform((self.sigma_u_min + self.sigma_u_max) / 2, self.sigma_u_max)
            self.sigma_v = random.uniform(self.sigma_v_min, (self.sigma_v_min + self.sigma_v_max) / 2)
        u_filter = VonKarmanFilter(self.L_u, self.sigma_u, self.V, self.dt)
        v_filter = VonKarmanFilter(self.L_v, self.sigma_v, self.V, self.dt)
        return u_filter, v_filter

    def __call__(self, rho, speed, d_cp_cg, **kwargs):
        if self.prev_rho is None:
            self.prev_rho = rho
        elif rho != self.prev_rho:
            np.random.set_state(self._initial_np_rng_state)
            self.u_filter.state = self._initial_u_state.copy()
            self.v_filter.state = self._initial_v_state.copy()
            self.prev_rho = rho
        gust_u = self.u_filter.step()
        gust_v = self.v_filter.step()
        dF = 0.5 * rho * speed * np.array([gust_u, gust_v]) * self.frontal_area
        dM = dF[1] * d_cp_cg
        return dF, dM

    def reset(self):
        np.random.seed(None)
        self.u_filter, self.v_filter = self.create_random_disturbance()
        self._initial_np_rng_state = np.random.get_state()
        self._initial_u_state = self.u_filter.state.copy()
        self._initial_v_state = self.v_filter.state.copy()
        self.prev_rho = None