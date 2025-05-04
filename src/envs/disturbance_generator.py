import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    def __init__(self, dt: float, V: float):
        self.L_u_min, self.L_u_max = 100.0, 500.0
        self.L_v_min, self.L_v_max = 30.0, 300.0
        self.sigma_u_min, self.sigma_u_max = 0.5, 2.0
        self.sigma_v_min, self.sigma_v_max = 0.5, 2.0
        self.V = V
        self.dt = dt
        self.u_filter, self.v_filter = self._new_filters()

        # Logging
        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': []}


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

    def __call__(self, rho: float):        
        # generate gusts
        gust_u = self.u_filter.step()
        gust_v = self.v_filter.step()
        self.log_data['gust_u'].append(gust_u)
        self.log_data['gust_v'].append(gust_v)
        return gust_u, gust_v

    def reset(self):
        # reseed and regenerate filters
        np.random.seed(None)
        self.u_filter, self.v_filter = self._new_filters()
        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': []}
        
    def plot_disturbance_generator(self, save_path):
        time = np.arange(0, len(self.log_data['gust_u'])) * self.dt

        plt.figure(figsize=(20,15))
        plt.suptitle('Von Kármán Disturbance Generator', fontsize=24)
        gs = gridspec.GridSpec(1,2, width_ratios=[1,1])

        ax1 = plt.subplot(gs[0])
        ax1.plot(time, self.log_data['gust_v'], linewidth=4, color='blue', label='Vertical')
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel('Gust [m/s]', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.set_title('Vertical', fontsize=20)
        ax1.grid(True)

        ax2 = plt.subplot(gs[1])
        ax2.plot(time, self.log_data['gust_u'], linewidth=4, color='blue', label='Horizontal')
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_ylabel('', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_title('Horizontal', fontsize=20)
        ax2.grid(True)
        plt.savefig(save_path + 'VonKarmenDisturbanceGenerator.png')
        plt.close()

        

def compile_disturbance_generator(dt : float,
                                  flight_phase : str):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
    if flight_phase == 'subsonic':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/subsonic/trajectory.csv')
    elif flight_phase == 'supersonic':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/supersonic/trajectory.csv')
    elif flight_phase == 'flip_over_boostbackburn':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/flip_over_boostbackburn/trajectory.csv')
    elif flight_phase == 'ballistic_arc_descent':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/ballistic_arc_descent/trajectory.csv')
    elif flight_phase == 're_entry_burn':
        data = pd.read_csv('data/agent_saves/SupervisoryLearning/re_entry_burn/trajectory.csv')
    else:
        raise ValueError(f"Flight phase {flight_phase} not supported")
    
    mean_vy = data['vy[m/s]'].median()
    mean_vx = data['vx[m/s]'].median()
    V = np.sqrt(mean_vx**2 + mean_vy**2)
    print('V: ', V)
    return VKDisturbanceGenerator(dt, V)



def test_disturbance_generator_subsonic():
    dt = 0.01
    flight_phase = 'subsonic'
    disturbance_generator = compile_disturbance_generator(dt, flight_phase)
    from utils.atmosphere_dynamics import endo_atmospheric_model
    data = pd.read_csv('data/agent_saves/SupervisoryLearning/subsonic/trajectory.csv')
    altitudes = data['y[m]'].to_numpy()
    for altitude in altitudes:
        rho, _, _ = endo_atmospheric_model(altitude)
        disturbance_generator(float(rho))
        print('Density: ', rho, 'Gust: ', disturbance_generator.log_data['gust_v'][-1])
    disturbance_generator.plot_disturbance_generator('results/verification/disturbance_generator_verification/')

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    print(sys.path)
    test_disturbance_generator_subsonic()



