import csv
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
    def __init__(self, dt: float, V: float, frontal_area: float, type: str, flight_phase: str):
        # TODO:V should be vertical speed of rocket mean, during that flight phase, estimate from the control/supervisory data?
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

        self.save_path = f'results/{type}/{flight_phase}'

        # Logging
        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': [],
            'dF_x': [],
            'dF_y': [],
            'dM': []}


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

        self.log_data['gust_u'].append(gust_u)
        self.log_data['gust_v'].append(gust_v)
        self.log_data['dF_x'].append(dF[0])
        self.log_data['dF_y'].append(dF[1])
        self.log_data['dM'].append(dM)
        return dF, dM

    def reset(self):
        # reseed and regenerate filters
        np.random.seed(None)
        self.u_filter, self.v_filter = self._new_filters()
        self._rng_state = np.random.get_state()
        self._u0 = self.u_filter.state.copy()
        self._v0 = self.v_filter.state.copy()
        self.prev_rho = None
        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': [],
            'dF_x': [],
            'dF_y': [],
            'dM': []}
        
    def plot_disturbance_generator(self):
        time = np.arange(0, len(self.log_data['gust_u'])) * self.dt

        plt.figure(figsize=(20,15))
        plt.suptitle('Von Kármán Disturbance Generator', fontsize=24)
        gs = gridspec.GridSpec(2,2, height_ratios=[3,1], width_ratios=[1,1])

        ax1 = plt.subplot(gs[0,0])
        ax1.plot(time, self.log_data['gust_v'], linewidth=4, color='purple', label='Vertical')
        ax1.plot(time, self.log_data['gust_u'], linewidth=4, color='green', label='Horizontal')
        ax1.legend(fontsize=20)
        ax1.set_xlabel('Time [s]', fontsize=20)
        ax1.set_ylabel('Gust [m/s]', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.set_title('Vertical Gust', fontsize=20)
        ax1.grid(True)

        ax2 = plt.subplot(gs[0,1])
        if max(self.log_data['dM']) > 1e6:
            ax2.plot(time, np.array(self.log_data['dM'])/1e6, linewidth=4, color='blue')
            ax2.set_ylabel('Moment [MNm]', fontsize=20)
        elif max(self.log_data['dM']) > 1e3:
            ax2.plot(time, np.array(self.log_data['dM'])/1e3, linewidth=4, color='blue')
            ax2.set_ylabel('Moment [kNm]', fontsize=20)
        else:
            ax2.plot(time, np.array(self.log_data['dM']), linewidth=4, color='blue')
            ax2.set_ylabel('Moment [Nm]', fontsize=20)
        ax2.set_xlabel('Time [s]', fontsize=20)
        ax2.set_title('Moment', fontsize=20)
        ax2.grid(True)

        ax3 = plt.subplot(gs[1,1])
        if max(self.log_data['dF_x']) > 1e6:
            ax3.plot(time, np.array(self.log_data['dF_x'])/1e6, linewidth=4, color='blue')
            ax3.set_ylabel('Force [MN]', fontsize=20)
        elif max(self.log_data['dF_x']) > 1e3:
            ax3.plot(time, np.array(self.log_data['dF_x'])/1e3, linewidth=4, color='blue')
            ax3.set_ylabel('Force [kN]', fontsize=20)
        else:
            ax3.plot(time, np.array(self.log_data['dF_x']), linewidth=4, color='blue')
            ax3.set_ylabel('Force [N]', fontsize=20)
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.set_title('Horizontal Force', fontsize=20)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1,0])
        if max(self.log_data['dF_y']) > 1e6:
            ax4.plot(time, np.array(self.log_data['dF_y'])/1e6, linewidth=4, color='blue')
            ax4.set_ylabel('Force [MN]', fontsize=20)
        elif max(self.log_data['dF_y']) > 1e3:
            ax4.plot(time, np.array(self.log_data['dF_y'])/1e3, linewidth=4, color='blue')
            ax4.set_ylabel('Force [kN]', fontsize=20)
        else:
            ax4.plot(time, np.array(self.log_data['dF_y']), linewidth=4, color='blue')
            ax4.set_ylabel('Force [N]', fontsize=20)
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.set_title('Vertical Force', fontsize=20)
        ax4.grid(True)

        plt.savefig(self.save_path + 'VonKarmenDisturbanceGenerator.png')
        plt.close()

        

def compile_disturbance_generator(dt : float,
                                  type : str,
                                  flight_phase : str):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]
    frontal_area = float(sizing_results['Rocket frontal area'])  # m²
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
    
    mean_vy = data['vy[m/s]'].mean()
    mean_vx = data['vx[m/s]'].mean()
    V = np.sqrt(mean_vx**2 + mean_vy**2)
    return VKDisturbanceGenerator(dt, V, frontal_area, type, flight_phase)