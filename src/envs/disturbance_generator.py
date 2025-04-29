import numpy as np
import math
import random
from scipy.signal import cont2discrete

class VonKarmanFilter:
    """Second-order shaping filter for Von Kármán gust according to standard model."""
    def __init__(self, L: float, sigma: float, V: float, dt: float):
        self.L = L
        self.sigma = sigma
        self.V = V
        self.dt = dt
        self._init_filter()

    def _init_filter(self):
        omega0 = self.V / self.L
        zeta = 1.0 / math.sqrt(2.0)
        scale = math.sqrt(math.pi / (2.0 * omega0**3))
        A_c = np.array([[0.0, 1.0],
                        [-omega0**2, -2.0 * zeta * omega0]])
        B_c = np.array([[0.0],
                        [self.sigma * scale]])
        C_c = np.array([[0.0, 1.0]])
        D_c = np.zeros((1, 1))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C_c, D_c), self.dt)
        self.Ad = A_d
        self.Bd = B_d.flatten()
        self.Cd = C_d
        self.state = np.zeros(2)

    def step(self) -> float:
        w = np.random.randn()
        self.state = self.Ad @ self.state + self.Bd * w
        return float(self.Cd @ self.state)

    def reset(self):
        self._init_filter()

class VKDisturbanceGenerator:
    """Generates body-axis disturbance forces and pitching moment using Von Kármán filters."""
    def __init__(self, dt: float, V: float, frontal_area: float):
        # parameter ranges
        self.L_u_min, self.L_u_max = 100.0, 500.0
        self.L_v_min, self.L_v_max = 30.0, 300.0
        self.sigma_u_min, self.sigma_u_max = 0.5, 2.0
        self.sigma_v_min, self.sigma_v_max = 0.5, 2.0
        self.frontal_area = frontal_area
        self.V = V
        self.dt = dt
        # capture initial RNG and filter states for density scaling tests
        self._rng_state = np.random.get_state()
        self.u_filter, self.v_filter = self._new_filters()
        self._u0 = self.u_filter.state.copy()
        self._v0 = self.v_filter.state.copy()
        self.prev_rho = None

    def _new_filters(self):
        # sample length scales
        if random.random() < 0.5:
            L_u = random.uniform(self.L_u_min, (self.L_u_min + self.L_u_max) / 2)
            L_v = random.uniform((self.L_v_min + self.L_v_max) / 2, self.L_v_max)
        else:
            L_u = random.uniform((self.L_u_min + self.L_u_max) / 2, self.L_u_max)
            L_v = random.uniform(self.L_v_min, (self.L_v_min + self.L_v_max) / 2)
        # sample noise intensities
        if random.random() < 0.5:
            sigma_u = random.uniform(self.sigma_u_min, (self.sigma_u_min + self.sigma_u_max) / 2)
            sigma_v = random.uniform((self.sigma_v_min + self.sigma_v_max) / 2, self.sigma_v_max)
        else:
            sigma_u = random.uniform((self.sigma_u_min + self.sigma_u_max) / 2, self.sigma_u_max)
            sigma_v = random.uniform(self.sigma_v_min, (self.sigma_v_min + self.sigma_v_max) / 2)
        # create filters
        u = VonKarmanFilter(L_u, sigma_u, self.V, self.dt)
        v = VonKarmanFilter(L_v, sigma_v, self.V, self.dt)
        # store parameters
        self.L_u, self.L_v = L_u, L_v
        self.sigma_u, self.sigma_v = sigma_u, sigma_v
        return u, v

    def __call__(self, rho: float, speed: float, d_cp_cg: float):
        # repeat same gust for density changes
        if self.prev_rho is None:
            self.prev_rho = rho
        elif rho != self.prev_rho:
            np.random.set_state(self._rng_state)
            self.u_filter.state = self._u0.copy()
            self.v_filter.state = self._v0.copy()
            self.prev_rho = rho
        # generate gusts
        gust_u = self.u_filter.step()
        gust_v = self.v_filter.step()
        # compute forces and moment
        dF = 0.5 * rho * speed * np.array([gust_u, gust_v]) * self.frontal_area
        dM = dF[1] * d_cp_cg
        return dF, dM

    def reset(self):
        # reseed and regenerate filters
        np.random.seed(None)
        self.u_filter, self.v_filter = self._new_filters()
        self._rng_state = np.random.get_state()
        self._u0 = self.u_filter.state.copy()
        self._v0 = self.v_filter.state.copy()
        self.prev_rho = None
