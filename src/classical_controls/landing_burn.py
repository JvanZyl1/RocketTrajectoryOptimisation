import csv
import pandas as pd
from src.classical_controls.landing_burn_optimise import LandingBurnOptimiser
from src.envs.utils.atmosphere_dynamics import gravity_model_endo, endo_atmospheric_model

class LandingBurn:
    def __init__(self, optimise_bool = False):
        self.dt = 0.1
        self.minimum_throttle = 0.4
        self.load_params()
        if optimise_bool:
            self.optimiser = LandingBurnOptimiser()
            self.optimiser()

    def initialise_logging(self):
        pass

    def initialise_conditions(self):
        pass

    def reset(self):
        pass

    def closed_loop_step(self):
        pass

    def save_results(self):
        pass

    def plot_results(self):
        pass

    def run_closed_loop(self):
        pass
    
    
