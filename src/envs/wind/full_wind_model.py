import numpy as np
from src.envs.wind.vonkarman import VKDisturbanceGenerator
from src.envs.wind.HorizontalWindSpeed import compile_horizontal_fixed_wind

class WindModel:
    def __init__(self, dt : float):
        self.dt = dt
        self.compile_von_karman_generator()
        self.V_VK = 100 # m/s
        self.VK_y_threshold = 15000 # m

    def compile_von_karman_generator(self):
        self.von_karman_generator_class = VKDisturbanceGenerator(self.dt, self.V_VK)

    def compile_horizontal_fixed_wind(self):
        random_percentile = float(np.random.randint(50, 99))
        self.horizontal_fixed_wind_func = compile_horizontal_fixed_wind(random_percentile)

    def __call__(self, y : float):
        fixed_vx = self.horizontal_fixed_wind_func(y)
        if y < self.VK_y_threshold:
            vx_vk, vy_vk = self.von_karman_generator_class()
        else:
            vx_vk, vy_vk = 0, 0
        return fixed_vx + vx_vk, vy_vk
    
    def reset(self):
        self.von_karman_generator_class.reset()
        self.compile_horizontal_fixed_wind()