import numpy as np
from src.envs.wind.vonkarman import VKDisturbanceGenerator
from src.envs.wind.HorizontalWindSpeed import compile_horizontal_fixed_wind
import matplotlib.pyplot as plt

class WindModel:
    def __init__(self, dt : float):
        self.dt = dt
        self.V_VK = 100 # m/s
        self.VK_y_threshold = 15000 # m
        self.compile_von_karman_generator()
        self.compile_horizontal_fixed_wind()
        self.vx_vals = []
        self.uy_vals = []

    def compile_von_karman_generator(self):
        self.von_karman_generator_class= VKDisturbanceGenerator(self.dt, self.V_VK)
        dict_von_karman_generator_class = self.von_karman_generator_class.return_dict_characteristics()
        # Round to 3 significant digits
        self.L_u = round(dict_von_karman_generator_class['L_u'], 0)
        self.L_v = round(dict_von_karman_generator_class['L_v'], 0)
        self.sigma_u = round(dict_von_karman_generator_class['sigma_u'], 2)
        self.sigma_v = round(dict_von_karman_generator_class['sigma_v'], 2)

    def compile_horizontal_fixed_wind(self):
        self.random_percentile = float(np.random.randint(50, 99))
        self.horizontal_fixed_wind_func = compile_horizontal_fixed_wind(self.random_percentile)

    def __call__(self, y : float):
        fixed_vx = self.horizontal_fixed_wind_func(y)
        if y < self.VK_y_threshold:
            vx_vk, vy_vk = self.von_karman_generator_class()
        else:
            vx_vk, vy_vk = 0, 0
        self.vx_vals.append(fixed_vx + vx_vk)
        self.uy_vals.append(vy_vk)
        return fixed_vx + vx_vk, vy_vk
    
    def reset(self):
        self.von_karman_generator_class.reset()
        self.compile_horizontal_fixed_wind()
        self.vx_vals = []
        self.uy_vals = []

    def plot_wind_model(self, save_path):
        time = np.arange(0, len(self.vx_vals)) * self.dt
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Wind Model - {self.random_percentile} percentile horizontal wind speed, \n'
                     f'L_u = {self.L_u} m, L_v = {self.L_v} m, \n'
                     f'sigma_u = {self.sigma_u} m/s, sigma_v = {self.sigma_v} m/s', fontsize = 16)
        axs[0].plot(time, self.vx_vals, color='blue', linewidth = 4)
        axs[0].set_xlabel('Time (spanned) [s]', fontsize = 20)
        axs[0].set_ylabel('Wind Speed [m/s]', fontsize = 20)
        axs[0].set_title('Horizontal', fontsize = 22)
        axs[0].grid(True)
        axs[0].tick_params(axis='both', which='major', labelsize=16)
        
        # Plot vertical wind component
        axs[1].plot(time, self.uy_vals, color='red', linewidth = 4)
        axs[1].set_xlabel('Time (spanned) [s]', fontsize = 20)
        axs[1].set_ylabel('Wind Speed [m/s]', fontsize = 20)
        axs[1].set_title('Vertical', fontsize = 22)
        axs[1].grid(True)
        axs[1].tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        