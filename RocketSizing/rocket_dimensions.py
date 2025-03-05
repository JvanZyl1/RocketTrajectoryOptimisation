import math
import numpy as np

#### 1) TANK SIZING ####
def cylindrical_tank_dimensions(mass: float,
                                density: float,
                                radius: float):
    volume = mass / density
    height = volume / (math.pi * radius**2)
    return height, volume

def fuel_to_oxidiser_mass_calculator(fuel_mass_required: float,
                                     fuel_to_oxidiser_ratio: float) -> float:
    oxidiser_mass = fuel_mass_required / fuel_to_oxidiser_ratio
    fuel_mass = fuel_mass_required - oxidiser_mass
    return oxidiser_mass, fuel_mass

def tank_sizing_constant_radius(propellant_mass: float,
                                density_oxidiser : float,
                                density_fuel : float,
                                fuel_to_oxidiser_ratio : float,
                                tank_radius : float):    
    oxidiser_mass, fuel_mass = fuel_to_oxidiser_mass_calculator(propellant_mass, fuel_to_oxidiser_ratio)
    fuel_tank_height, _ = cylindrical_tank_dimensions(fuel_mass, density_fuel, tank_radius)
    oxidiser_tank_height, _ = cylindrical_tank_dimensions(oxidiser_mass, density_oxidiser, tank_radius)
    return oxidiser_tank_height, fuel_tank_height, oxidiser_mass, fuel_mass



class rocket_dimensions:
    def __init__(self,
                 tank_wall_thickness: float,
                 rocket_radius: float):
        self.wall_thickness_tanks = tank_wall_thickness
        self.rocket_radius = rocket_radius
    

    def size_tanks(self,
                   propellant_mass_stage_1,
                   propellant_mass_stage_2):
        density_LOX = 1200      #[kg/m^3] : Oxidiser density
        density_LCH4 = 450      #[kg/m^3] : Fuel density
        oxidiser_to_fuel_ratio = 3.545

        tank_radius = self.rocket_radius - self.wall_thickness_tanks
        tank_sizing_constant_radius_lambda_func = lambda propellant_mass : tank_sizing_constant_radius(propellant_mass,
                                                                                                    density_LOX,
                                                                                                    density_LCH4,
                                                                                                    oxidiser_to_fuel_ratio,
                                                                                                    tank_radius)
        
        oxidiser_tank_height_stage_1, fuel_tank_height_stage_1, oxidiser_mass_stage_1, fuel_mass_stage_1 = tank_sizing_constant_radius_lambda_func(propellant_mass_stage_1)
        oxidiser_tank_height_stage_2, fuel_tank_height_stage_2, oxidiser_mass_stage_2, fuel_mass_stage_2 = tank_sizing_constant_radius_lambda_func(propellant_mass_stage_2)


    def inertia_and_x_cog_creator(self):
        stage_1_engines_mass = sum(number_of_engines_per_ring_stage_1) * config['engine_dictionaries'][0]['full_integrated_weight']
        stage_2_engines_mass = number_of_engines_stage_2 * config['engine_dictionaries'][1]['full_integrated_weight']

        x_cog_inertia_subrocket_0_lambda, x_cog_inertia_subrocket_1_lambda, subrocket_lengths, x_cog_payload = \
        inertia_funcs_creator(structural_masses=[structural_mass_stage_1, structural_mass_stage_2],
                              engine_masses=[stage_1_engines_mass, stage_2_engines_mass],
                              oxidiser_tank_heights=[oxidiser_tank_height_1, oxidiser_tank_height_2],
                              fuel_tank_heights=[fuel_tank_height_1, fuel_tank_height_2],
                              oxidiser_masses=[oxidiser_mass_stage_1, oxidisier_mass_stage_2],
                              fuel_masses=[fuel_mass_stage_1, fuel_mass_stage_2],
                              engine_heights=[config['engine_dictionaries'][0]['engine_height'],
                                              config['engine_dictionaries'][1]['engine_height']],
                              payload_mass=config['payload_mass'],
                              wall_thickness_tanks=wall_thickness_tanks,
                              rocket_radius=rocket_radius,
                              print_bool=print_bool)
        return 
        

