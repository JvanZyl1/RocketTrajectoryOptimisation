import math
import numpy as np
import csv

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


#### 2) INERTIA CALCULATIONS ####

def cylinder_inertia_pitch(mass, length, displacement):
    inertia = 1/12 * mass * length**2 + mass * displacement**2
    return inertia

def propellant_inertia_calculator(oxidiser_initial_mass,
                                  fuel_initial_mass,
                                  oxidiser_tank_height,
                                  fuel_tank_height,
                                  fill_level,
                                  lower_section_height):
    if fill_level == 1.0:
        fill_level = 0.9999999999999999

    # Around propellant cog
    h_ox = oxidiser_tank_height * fill_level
    h_fuel = fuel_tank_height * fill_level

    x_ox = lower_section_height + fuel_tank_height + h_ox/2
    x_fuel = lower_section_height + h_fuel/2

    m_ox = oxidiser_initial_mass * fill_level
    m_fuel = fuel_initial_mass * fill_level
    x_cog_prop = (m_fuel * x_fuel + m_ox * x_ox) / (m_fuel + m_ox)

    d_ox = x_ox - x_cog_prop
    I_ox = cylinder_inertia_pitch(m_ox, h_ox, d_ox)

    d_fuel = x_cog_prop - x_fuel
    I_fuel = cylinder_inertia_pitch(m_fuel, h_fuel, d_fuel)

    I_prop = I_ox + I_fuel

    return I_prop, x_cog_prop

def rocket_stage_inertia_prop_and_dry(I_dry: float,
                                      I_prop: float,
                                      x_cog_dry: float,
                                      x_cog_prop: float,
                                      m_dry: float,
                                      m_prop: float):
    
    x_cog_wet = (m_dry * x_cog_dry + m_prop * x_cog_prop) / (m_dry + m_prop)

    # Parallel_axis_theorem
    d_dry = x_cog_dry - x_cog_wet
    I_dry = I_dry + m_dry * d_dry**2

    d_prop = x_cog_prop - x_cog_wet
    I_prop = I_prop + m_prop * d_prop**2

    I_stage = I_dry + I_prop

    return I_stage, x_cog_wet

def find_stage_inertia_lambda_func_creation(oxidisier_mass,
             fuel_mass,
             oxidiser_tank_height,
             fuel_tank_height,
             lower_section_height,
             I_dry,
             x_cog_dry,
             structural_mass):
    def func(oxidisier_mass,
             fuel_mass,
             oxidiser_tank_height,
             fuel_tank_height,
             fill_level,
             lower_section_height,
             I_dry,
             x_cog_dry,
             structural_mass):
        I_prop, x_cog_prop = propellant_inertia_calculator(oxidisier_mass,
                                                        fuel_mass,
                                                        oxidiser_tank_height,
                                                        fuel_tank_height,
                                                        fill_level,
                                                        lower_section_height)
        
        I_wet, x_cog_wet = rocket_stage_inertia_prop_and_dry(I_dry,
                                                            I_prop,
                                                            x_cog_dry,
                                                            x_cog_prop,
                                                            structural_mass,
                                                            oxidisier_mass + fuel_mass)
        return I_wet, x_cog_wet
    
    stage_inertia_lambda_func = lambda fill_level: func(oxidisier_mass,
                                                              fuel_mass,
                                                              oxidiser_tank_height,
                                                              fuel_tank_height,
                                                              fill_level,
                                                              lower_section_height,
                                                              I_dry,
                                                              x_cog_dry,
                                                              structural_mass)
        
    return stage_inertia_lambda_func


def dry_inertia_stages(x_cog_dry: float,
                            engine_mass_actual: float,
                            engine_height: float,
                            lower_mass_actual: float,
                            lower_section_height: float,
                            tank_strc_mass_actual: float,
                            total_tank_height: float,
                            upper_mass_actual: float,
                            upper_section_height: float):
    # Calculate dry mass inertia
    x_engine_dry = -(x_cog_dry+engine_height/2)
    I_engine_dry = cylinder_inertia_pitch(engine_mass_actual, engine_height, x_engine_dry)
    x_lower_dry = -(x_cog_dry-lower_section_height/2)
    I_lower_dry = cylinder_inertia_pitch(lower_mass_actual, lower_section_height, x_lower_dry)
    x_tank_strc_dry = -(x_cog_dry - (lower_section_height + total_tank_height/2))
    I_tank_strc_dry = cylinder_inertia_pitch(tank_strc_mass_actual, total_tank_height, x_tank_strc_dry)
    x_upper_dry = -(x_cog_dry - (lower_section_height + total_tank_height + upper_section_height/2))
    I_upper_dry = cylinder_inertia_pitch(upper_mass_actual, upper_section_height, x_upper_dry)
    I_dry_stage = I_engine_dry + I_lower_dry + I_tank_strc_dry + I_upper_dry
    return I_dry_stage

def rocket_section_sizing_first_stage(structural_mass_stage_1: float,
                                      engine_mass_stage_1: float,
                                      oxidiser_tank_height_stage_1: float,
                                      fuel_tank_height_stage_1: float,
                                      wall_thickness: float,
                                      rocket_radius: float,
                                      engine_height: float,
                                      fuel_mass: float,
                                      oxidiser_mass: float,
                                      upper_lower_ratio: float = 1  # upper section mas * X : lower section mass
                                      ):
    # Densities
    rho_stainless_steel = 8000 # kg/m^3 : 304L
    rho_tank_strc = rho_stainless_steel # kg/m^3
    rho_sections = np.array([rho_stainless_steel, rho_stainless_steel]) # kg/m^3
    # Mass of structural tanks
    mass_tank_strc = math.pi * (oxidiser_tank_height_stage_1 + fuel_tank_height_stage_1) * (rocket_radius**2 - (rocket_radius - wall_thickness)**2) * rho_tank_strc
    # Mass and volumes of sections
    sections_mass = (structural_mass_stage_1 - engine_mass_stage_1 - mass_tank_strc) * np.array([1/(1+upper_lower_ratio), upper_lower_ratio/(1+upper_lower_ratio)])
    sections_volume = sections_mass / rho_sections
    section_heights = sections_volume / (math.pi * rocket_radius**2)
    assert mass_tank_strc + engine_mass_stage_1 + sections_mass.sum() == structural_mass_stage_1, "Masses do not add up"
    # Cog of dry
    x_engine = -engine_height/2
    x_lower = section_heights[0]/2
    x_tank_strc = (section_heights[0] + (oxidiser_tank_height_stage_1 + fuel_tank_height_stage_1)/2)
    x_upper = (section_heights[0] + oxidiser_tank_height_stage_1 + fuel_tank_height_stage_1 + section_heights[1]/2)
    x_cog_dry = (engine_mass_stage_1 * x_engine + sections_mass[0] * x_lower + mass_tank_strc * x_tank_strc + sections_mass[1] * x_upper) / structural_mass_stage_1
    
    # Now for propellant : fuel tank then oxidiser tank
    x_fuel = section_heights[0] + fuel_tank_height_stage_1/2
    x_oxidiser = section_heights[0] + fuel_tank_height_stage_1 + oxidiser_tank_height_stage_1/2
    x_cog_prop = (fuel_mass * x_fuel + oxidiser_mass * x_oxidiser) / (fuel_mass + oxidiser_mass)

    # Total wet
    x_cog_wet = ( x_cog_dry * structural_mass_stage_1 + x_cog_prop * (fuel_mass + oxidiser_mass)) / (structural_mass_stage_1 + fuel_mass + oxidiser_mass)

    # Total rocket height
    total_rocket_height = section_heights.sum() + oxidiser_tank_height_stage_1 + fuel_tank_height_stage_1

    # Inertia calculations
    I_dry_stage_1 = dry_inertia_stages(x_cog_dry,
                                             engine_mass_stage_1,
                                                engine_height,
                                                sections_mass[0],
                                                section_heights[0],
                                                mass_tank_strc,
                                                oxidiser_tank_height_stage_1 + fuel_tank_height_stage_1,
                                                sections_mass[1],
                                                section_heights[1])
    
    stage_1_inertia_lambda_func =  find_stage_inertia_lambda_func_creation(oxidiser_mass,
                                                                            fuel_mass,
                                                                            oxidiser_tank_height_stage_1,
                                                                            fuel_tank_height_stage_1,
                                                                            section_heights[0],
                                                                            I_dry_stage_1,
                                                                            x_cog_dry,
                                                                            structural_mass_stage_1)
    
    return (x_cog_dry, x_cog_prop, x_cog_wet, total_rocket_height, section_heights, I_dry_stage_1, stage_1_inertia_lambda_func)

def rocket_section_sizing_second_stage(structural_mass_stage_2 : float,
                                       fairing_mass: float,
                                       engine_mass_stage_2: float,
                                       oxidiser_tank_height_stage_2: float,
                                       fuel_tank_height_stage_2: float,
                                       wall_thickness: float,
                                       rocket_radius: float,
                                       engine_height: float,
                                       fuel_mass: float,
                                       oxidiser_mass: float):
    # Structural mass without fairing
    structural_mass_no_fairing = structural_mass_stage_2 - fairing_mass

    #
    x_cog_dry_no_payload, x_cog_prop_no_payload, x_cog_wet_no_payload, stage_2_height_no_payload, \
          section_heights, I_dry_stage_2, stage_2_inertia_lambda_func = rocket_section_sizing_first_stage(structural_mass_no_fairing,
                                                                                                          engine_mass_stage_2,
                                                                                                          oxidiser_tank_height_stage_2,
                                                                                                          fuel_tank_height_stage_2,
                                                                                                          wall_thickness,
                                                                                                          rocket_radius,
                                                                                                          engine_height,
                                                                                                          fuel_mass,
                                                                                                          oxidiser_mass)
    
    I_prop_stage_2, x_cog_prop_no_payload_check = propellant_inertia_calculator(oxidiser_mass,
                                                                                fuel_mass,
                                                                                oxidiser_tank_height_stage_2,
                                                                                fuel_tank_height_stage_2,
                                                                                1,
                                                                                section_heights[0])

    assert math.isclose(x_cog_prop_no_payload, x_cog_prop_no_payload_check, rel_tol=1e-9), "Cog prop not equal"

    I_wet_stage_2, x_cog_wet_no_payload_check = rocket_stage_inertia_prop_and_dry(I_dry_stage_2,
                                                                                  I_prop_stage_2,
                                                                                  x_cog_dry_no_payload,
                                                                                  x_cog_prop_no_payload,
                                                                                  structural_mass_no_fairing,
                                                                                  oxidiser_mass + fuel_mass)

    assert math.isclose(x_cog_wet_no_payload, x_cog_wet_no_payload_check, rel_tol=1e-9), "Cog wet not equal"

    stage_2_inertia_lambda_func =  find_stage_inertia_lambda_func_creation(oxidiser_mass,
                                                                           fuel_mass,
                                                                           oxidiser_tank_height_stage_2,
                                                                           fuel_tank_height_stage_2,
                                                                           section_heights[0],
                                                                           I_dry_stage_2,
                                                                           x_cog_dry_no_payload,
                                                                           structural_mass_no_fairing)
    
    I_wet_CHECK, x_cog_wet_CHECK = stage_2_inertia_lambda_func(1)
    assert math.isclose(I_wet_stage_2, I_wet_CHECK, rel_tol=1e-9), "I wet not equal"
    
    return (x_cog_dry_no_payload, x_cog_prop_no_payload, x_cog_wet_no_payload, stage_2_height_no_payload, \
            section_heights, I_wet_stage_2, stage_2_inertia_lambda_func)

def parabolic_nose_volume(radius: float,
                          length: float):
    # Volume of a parabolic cone: V = pi * diameter^2 * h / 8
    volume = math.pi * (2*radius)**2 * length / 8
    return volume

def cone_moment_of_inertia(radius: float,
                            length: float,
                            mass: float,
                            x_cog: float):
    # Approximate of a parabolic cone
    # https://resources.wolframcloud.com/FormulaRepository/resources/Moment-of-Inertia-of-a-Cone
    inertia = 3/80 * mass * (length**2 + 4 * radius**2) + mass * x_cog**2
    return inertia


def nose_sizing(rocket_radius: float,
             payload_density: float,
             fairing_density: float,
             payload_mass: float,
             payload_fairing_thickness : float):
    payload_radius = rocket_radius - payload_fairing_thickness

    # Size nose cone: parabolic cone: https://www.grc.nasa.gov/WWW/K-12/BGP/volume.html
    # Volume of a parabolic cone: V = pi * diameter^2 * h / 8
    # h = 8V / (pi * diameter^2)
    payload_volume = payload_mass / payload_density
    payload_length = 8 * payload_volume / (math.pi * (2*payload_radius)**2)

    # Fairing volume
    nose_length = payload_length + payload_fairing_thickness
    nose_radius = rocket_radius
    nose_volume = parabolic_nose_volume(nose_radius, nose_length)
    fairing_volume = nose_volume - payload_volume

    # Fairing mass
    fairing_mass = fairing_volume * fairing_density

    # Payload cog : 2 * h/3
    x_cog_payload = 2 * payload_length / 3

    # Fairing cog
    x_cog_fairing = 2 * nose_length / 3

    # Nose cog
    x_cog_nose = (payload_mass * x_cog_payload + fairing_mass * x_cog_fairing) / (payload_mass + fairing_mass)

    # Nose inertia : approximately a parabolic cone, and uniform density
    I_nose = cone_moment_of_inertia(nose_radius, nose_length, fairing_mass + payload_mass, x_cog_nose)

    return nose_radius, nose_length, fairing_mass, x_cog_nose, x_cog_payload, I_nose

def cog_tank(tank_height,
             m_fluid_intial,
             fill_level,
             x_rocketbase_to_tankbase):
    # Fill level is a percentage of the tank height
    x_cog = fill_level * tank_height / 2 + x_rocketbase_to_tankbase
    m_fluid = fill_level * m_fluid_intial
    return x_cog, m_fluid

def subrocket_0_cog_inertia(stage_1_structural_mass: float,
                      x_cog_dry_stage_1: float,
                      stage_1_lower_section_height: float,
                      stage_1_fuel_tank_height: float,
                      stage_1_oxidiser_tank_height: float,
                      stage_1_initial_fuel_mass: float,
                      stage_1_initial_oxidiser_mass: float,
                      stage_2_mass_no_fairing: float,
                      nose_mass: float,
                      x_cog_wet_stage_2: float,
                      x_cog_nose: float,
                      stage_1_height: float,
                      stage_2_height: float,
                      stage_1_inertia_lambda_func: callable,
                      I_stage_2_wet: float,
                      I_nose: float,
                      fuel_consumption_perc: float = 0 # [0-1] of fuel/O2 consumed
                      ):
    # Stage 1 dry mass
    stage_1_dry_mass = stage_1_structural_mass + stage_2_mass_no_fairing + nose_mass

    # Fuel tank
    fill_level = 1 - fuel_consumption_perc
    x_fuel = stage_1_lower_section_height + stage_1_fuel_tank_height/2
    x_cog_fuel, m_fuel = cog_tank(stage_1_fuel_tank_height,
                                  stage_1_initial_fuel_mass,
                                  fill_level,
                                  x_fuel)

    x_ox = stage_1_lower_section_height + stage_1_fuel_tank_height + stage_1_oxidiser_tank_height/2
    x_cog_ox, m_ox = cog_tank(stage_1_oxidiser_tank_height,
                              stage_1_initial_oxidiser_mass,
                              fill_level,
                              x_ox)

    # Adjust other cogs
    x_cog_stage_2 = stage_1_height + x_cog_wet_stage_2
    x_cog_nose = stage_1_height + stage_2_height + x_cog_nose

    # Calculate cog
    x_cog = (stage_1_dry_mass * x_cog_dry_stage_1 + \
             m_fuel * x_cog_fuel + \
             m_ox * x_cog_ox + \
             stage_2_mass_no_fairing * x_cog_stage_2 + \
             nose_mass * x_cog_nose) / \
             (stage_1_dry_mass + m_fuel + m_ox + stage_2_mass_no_fairing + nose_mass)
    
    # Inertia stuffs
    I_stage_1_not_in_right_axis, x_cog_stage_1 = stage_1_inertia_lambda_func(fill_level)
    d_stage_1 = x_cog - x_cog_stage_1
    d_stage_2 = x_cog - x_cog_stage_2
    d_nose = x_cog - x_cog_nose
    I_stage_1 = I_stage_1_not_in_right_axis + stage_1_dry_mass * d_stage_1**2
    I_stage_2_wet = I_stage_2_wet + stage_2_mass_no_fairing * d_stage_2**2
    I_nose = I_nose + nose_mass * d_nose**2

    inertia = I_stage_1 + I_stage_2_wet + I_nose
    
    return x_cog, inertia

def subrocket_1_cog_inertia(stage_2_structural_mass: float,
                x_cog_dry_stage_2: float, # NO PAYLOAD STAGE 2
                stage_2_lower_section_height: float,
                stage_2_fuel_tank_height: float,
                stage_2_oxidiser_tank_height: float,
                stage_2_initial_fuel_mass: float,
                stage_2_initial_oxidiser_mass: float,
                nose_mass: float,
                x_cog_nose: float,
                stage_2_height: float,
                stage_2_inertia_lambda_func: callable,
                inertia_nose: float,
                fuel_consumption_perc: float = 0 # [0-1] of fuel/O2 consumed
                ):
    # Stage 1 dry mass
    stage_2_dry_mass = stage_2_structural_mass + nose_mass

    # Fuel tank
    fill_level = 1 - fuel_consumption_perc
    x_fuel = stage_2_lower_section_height + stage_2_fuel_tank_height/2
    x_cog_fuel, m_fuel = cog_tank(stage_2_fuel_tank_height,
                                  stage_2_initial_fuel_mass,
                                  fill_level,
                                  x_fuel)

    x_ox = stage_2_lower_section_height + stage_2_fuel_tank_height + stage_2_oxidiser_tank_height/2
    x_cog_ox, m_ox = cog_tank(stage_2_oxidiser_tank_height,
                              stage_2_initial_oxidiser_mass,
                              fill_level,
                              x_ox)

    # Adjust other cogs
    x_cog_nose = stage_2_height + x_cog_nose

    # Calculate cog
    x_cog = (stage_2_dry_mass * x_cog_dry_stage_2 + \
             m_fuel * x_cog_fuel + \
             m_ox * x_cog_ox + \
             nose_mass * x_cog_nose) / \
             (stage_2_dry_mass + m_fuel + m_ox + nose_mass)
    
    I_stage_2_not_in_right_axis, x_cog_stage_2 = stage_2_inertia_lambda_func(fill_level)
    d_stage_2 = x_cog - x_cog_stage_2
    d_nose = x_cog - x_cog_nose
    I_stage_2 = I_stage_2_not_in_right_axis + stage_2_dry_mass * d_stage_2**2
    I_nose = inertia_nose + nose_mass * d_nose**2

    inertia = I_stage_2 + I_nose
    
    return x_cog, inertia

def d_cg_thrusters(x_cog,
                   engine_height):
    # Thrusters are at the bottom of the rocket
    d_cg_thrusters = x_cog + engine_height
    return d_cg_thrusters


class rocket_dimensions:
    def __init__(self,
                 rocket_radius: float,
                 propellant_masses: list,
                 structural_masses: list,
                 payload_mass: float,
                 number_of_engines: list):
        self.wall_thickness_tanks = 0.01
        self.rocket_radius = rocket_radius
        self.propellant_mass_stage_1 = propellant_masses[0]
        self.propellant_mass_stage_2 = propellant_masses[1]
        self.structural_mass_stage_1 = structural_masses[0]
        self.structural_mass_stage_2 = structural_masses[1]
        self.payload_mass = payload_mass
        self.number_of_engines_stage_1 = number_of_engines[0]
        self.number_of_engines_stage_2 = number_of_engines[1]

        self.engine_integrated_weight = 1720                #[kg]
        self.engine_height = 3.1                            #[m]
        self.payload_fairing_thickness = 0.1                #[m]
        self.payload_density = 2000                         #[kg/m^3]
        self.fairing_density = 7986                         #[kg/m^3]

        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Wall thickness tanks ', 'mm', self.wall_thickness_tanks*1000])
            writer.writerow(['Payload fairing thickness ', 'cm', self.payload_fairing_thickness*100])
            writer.writerow(['Payload density ', 'kg/m^3', self.payload_density])
            writer.writerow(['Fairing density ', 'kg/m^3', self.fairing_density])    

    def size_tanks(self):
        density_LOX = 1200      #[kg/m^3] : Oxidiser density
        density_LCH4 = 450      #[kg/m^3] : Fuel density
        oxidiser_to_fuel_ratio = 3.545

        tank_radius = self.rocket_radius - self.wall_thickness_tanks
        tank_sizing_constant_radius_lambda_func = lambda propellant_mass : tank_sizing_constant_radius(propellant_mass,
                                                                                                    density_LOX,
                                                                                                    density_LCH4,
                                                                                                    oxidiser_to_fuel_ratio,
                                                                                                    tank_radius)
        
        oxidiser_tank_height_stage_1, fuel_tank_height_stage_1, oxidiser_mass_stage_1, fuel_mass_stage_1 = tank_sizing_constant_radius_lambda_func(self.propellant_mass_stage_1)
        oxidiser_tank_height_stage_2, fuel_tank_height_stage_2, oxidiser_mass_stage_2, fuel_mass_stage_2 = tank_sizing_constant_radius_lambda_func(self.propellant_mass_stage_2)

        oxidiser_tank_heights = [oxidiser_tank_height_stage_1, oxidiser_tank_height_stage_2]
        fuel_tank_heights = [fuel_tank_height_stage_1, fuel_tank_height_stage_2]
        oxidiser_masses = [oxidiser_mass_stage_1, oxidiser_mass_stage_2]
        fuel_masses = [fuel_mass_stage_1, fuel_mass_stage_2]

        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Oxidiser tank height stage 1 ', 'm', oxidiser_tank_heights[0]])
            writer.writerow(['Fuel tank height stage 1 ', 'm', fuel_tank_heights[0]])
            writer.writerow(['Oxidiser mass stage 1 ', 'kg', oxidiser_masses[0]])
            writer.writerow(['Fuel mass stage 1 ', 'kg', fuel_masses[0]])
            writer.writerow(['Oxidiser tank height stage 2 ', 'm', oxidiser_tank_heights[1]])
            writer.writerow(['Fuel tank height stage 2 ', 'm', fuel_tank_heights[1]])
            writer.writerow(['Oxidiser mass stage 2 ', 'kg', oxidiser_masses[1]])
            writer.writerow(['Fuel mass stage 2 ', 'kg', fuel_masses[1]])

        return oxidiser_tank_heights, fuel_tank_heights, oxidiser_masses, fuel_masses


    def __call__(self):
        self.stage_1_engines_mass = self.number_of_engines_stage_1 * self.engine_integrated_weight
        self.stage_2_engines_mass = self.number_of_engines_stage_2 * self.engine_integrated_weight

        oxidiser_tank_heights, fuel_tank_heights, oxidiser_masses, fuel_masses = self.size_tanks()

        # 1) Stage 1
        x_cog_dry_stage_1, x_cog_prop_stage_1, x_cog_wet_stage_1, \
            stage_1_height, section_heights_stage_1, I_dry_stage_1, stage_1_inertia_lambda_func = \
            rocket_section_sizing_first_stage(structural_mass_stage_1 = self.structural_mass_stage_1,
                                            engine_mass_stage_1 = self.stage_1_engines_mass,
                                            oxidiser_tank_height_stage_1 = oxidiser_tank_heights[0],
                                            fuel_tank_height_stage_1 = fuel_tank_heights[0],
                                                wall_thickness = self.wall_thickness_tanks,
                                                rocket_radius = self.rocket_radius,
                                                engine_height = self.engine_height,
                                                fuel_mass = fuel_masses[0],
                                                oxidiser_mass = oxidiser_masses[0])
        
        # 2) Nose
        nose_radius, nose_length, fairing_mass, x_cog_nose, x_cog_payload, I_nose = \
            nose_sizing(rocket_radius = self.rocket_radius,
                        payload_density = self.payload_density,
                        fairing_density = self.fairing_density,                         # Stainless steel
                        payload_mass = self.payload_mass,
                        payload_fairing_thickness = self.payload_fairing_thickness)
        mass_nose = self.payload_mass + fairing_mass

        # 3) Stage 2
        x_cog_dry_stage_2_no_payload, x_cog_prop_stage_2_no_payload, x_cog_wet_stage_2_no_payload, stage_2_height_stage_2_no_payload, \
                section_heights_stage_2_no_payload, I_wet_stage_2_no_payload, stage_2_no_payload_inertia_lambda_func = \
                    rocket_section_sizing_second_stage(structural_mass_stage_2 = self.structural_mass_stage_2,
                                                    fairing_mass = fairing_mass,
                                                        engine_mass_stage_2 = self.stage_2_engines_mass,
                                                        oxidiser_tank_height_stage_2 = oxidiser_tank_heights[1],
                                                        fuel_tank_height_stage_2 = fuel_tank_heights[1],
                                                        wall_thickness = self.wall_thickness_tanks,
                                                        rocket_radius = self.rocket_radius,
                                                        engine_height = self.engine_height,
                                                        fuel_mass = fuel_masses[1],
                                                        oxidiser_mass = oxidiser_masses[1])
        mass_stage_2_no_payload = self.structural_mass_stage_2 + oxidiser_masses[1] + fuel_masses[1] - fairing_mass

        # 4) Lengths
        length_of_subrocket_0 = stage_1_height + stage_2_height_stage_2_no_payload + nose_length
        length_of_subrocket_1 = stage_2_height_stage_2_no_payload + nose_length
        length_of_subrocket_2 = stage_1_height
        lengths = [length_of_subrocket_0, length_of_subrocket_1, length_of_subrocket_2]

        # 5) Subrocket 0: Stage 1 + Stage 2 + Payload (nose)
        x_cog_inertia_subrocket_0_lambda = lambda fuel_consumption_perc : subrocket_0_cog_inertia(stage_1_structural_mass = self.structural_mass_stage_1,
                                        x_cog_dry_stage_1 = x_cog_dry_stage_1,
                                        stage_1_lower_section_height = section_heights_stage_1[0],
                                        stage_1_fuel_tank_height = fuel_tank_heights[0],
                                        stage_1_oxidiser_tank_height = oxidiser_tank_heights[0],
                                        stage_1_initial_fuel_mass= fuel_masses[0],
                                        stage_1_initial_oxidiser_mass= oxidiser_masses[0],
                                        stage_2_mass_no_fairing= mass_stage_2_no_payload,
                                        nose_mass= mass_nose,
                                        x_cog_wet_stage_2= x_cog_wet_stage_2_no_payload,
                                        x_cog_nose= x_cog_nose,
                                        stage_1_height= stage_1_height,
                                        stage_2_height= stage_2_height_stage_2_no_payload,
                                        stage_1_inertia_lambda_func= stage_1_inertia_lambda_func,
                                        I_stage_2_wet= I_wet_stage_2_no_payload,
                                        I_nose= I_nose,
                                        fuel_consumption_perc = fuel_consumption_perc)    

        # 6) Subrocket 1: Stage 2 + Payload (nose)
        x_cog_inertia_subrocket_1_lambda = lambda fuel_consumption_perc : subrocket_1_cog_inertia(
            stage_2_structural_mass= self.structural_mass_stage_2,
            x_cog_dry_stage_2= x_cog_dry_stage_2_no_payload,
            stage_2_lower_section_height= section_heights_stage_2_no_payload[0],
            stage_2_fuel_tank_height= fuel_tank_heights[1],
            stage_2_oxidiser_tank_height= oxidiser_tank_heights[1],
            stage_2_initial_fuel_mass= fuel_masses[1],
            stage_2_initial_oxidiser_mass= oxidiser_masses[1],
            nose_mass= mass_nose,
            x_cog_nose= x_cog_nose,
            stage_2_height= stage_2_height_stage_2_no_payload,
            stage_2_inertia_lambda_func= stage_2_no_payload_inertia_lambda_func,
            inertia_nose= I_nose,
            fuel_consumption_perc= fuel_consumption_perc)
            
        # 7) Thruster arms
        d_cg_thrusters_subrocket_0_lambda = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)
        d_cg_thrusters_subrocket_1_lambda = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)
            
        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Stage 1 upper section height ', 'm', section_heights_stage_1[1]])
            writer.writerow(['Stage 1 lower section height ', 'm', section_heights_stage_1[0]])
            writer.writerow(['Stage 2 lower section height ', 'm', section_heights_stage_2_no_payload[0]])
            writer.writerow(['Stage 2 upper section height ', 'm', section_heights_stage_2_no_payload[1]])
            writer.writerow(['Stage 1 height ', 'm', stage_1_height])
            writer.writerow(['Stage 2 height ', 'm', stage_2_height_stage_2_no_payload])
            writer.writerow(['Length of subrocket 0 ', 'm', length_of_subrocket_0])
            writer.writerow(['Length of subrocket 1 ', 'm', length_of_subrocket_1])
            writer.writerow(['Length of nose ', 'm', nose_length])

        # subrocket_0 : stage_1 + stage_2 + payload (nose)
        # subrocket_1 : stage_2 + payload (nose)
        # subrocket_2 : stage_1
        x_cog_inertia_subrocket_2_lambda = lambda fill_level: tuple(reversed(stage_1_inertia_lambda_func(fill_level))) # Beun fix baby
        d_cg_thrusters_subrocket_2_lambda = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)

        return (x_cog_inertia_subrocket_0_lambda, x_cog_inertia_subrocket_1_lambda, lengths, x_cog_payload, \
                d_cg_thrusters_subrocket_0_lambda, d_cg_thrusters_subrocket_1_lambda, \
                    x_cog_inertia_subrocket_2_lambda, d_cg_thrusters_subrocket_2_lambda, stage_1_height)