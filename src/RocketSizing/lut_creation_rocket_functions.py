import os
import dill
import csv
import numpy as np

def generate_lut_rocket_functions():
    with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
        rocket_functions = dill.load(f)

    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]

    # Full rocket
    full_rocket_cop_func = rocket_functions['cop_subrocket_0_lambda']
    full_rocket_x_cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_0_lambda']
    full_rocket_d_cp_cg_func = rocket_functions['d_cg_thrusters_subrocket_0_lambda']

    # Stage 1
    stage_1_cop_func = rocket_functions['cop_subrocket_2_lambda']
    stage_1_x_cog_inertia_func = rocket_functions['x_cog_inertia_subrocket_2_lambda']
    stage_1_d_cp_cg_func = rocket_functions['d_cg_thrusters_subrocket_2_lambda']

    # Propellant mass of stage 1
    propellant_mass_stage_1 = float(sizing_results['Propellant mass stage 1 (ascent)']) * 1000 \
        + float(sizing_results['Propellant mass stage 1 (descent)']) * 1000  # [kg]

    propellant_masses = np.linspace(0, propellant_mass_stage_1, 750)

    x_cog_inertias_full_rocket = np.zeros(len(propellant_masses))
    d_cp_cgs_full_rocket = np.zeros(len(propellant_masses))
    inertias_full_rocket = np.zeros(len(propellant_masses))

    x_cog_inertias_stage_1 = np.zeros(len(propellant_masses))
    d_cp_cgs_stage_1 = np.zeros(len(propellant_masses))
    inertias_stage_1 = np.zeros(len(propellant_masses))
    for i, propellant_mass in enumerate(propellant_masses):
        chi = (propellant_mass_stage_1 - propellant_mass) / propellant_mass_stage_1
        x_cog_inertia_full_rocket, inertia_full_rocket = full_rocket_x_cog_inertia_func(1 - chi)
        d_cp_cg_full_rocket = full_rocket_d_cp_cg_func(chi)

        x_cog_inertias_full_rocket[i] = x_cog_inertia_full_rocket
        d_cp_cgs_full_rocket[i] = d_cp_cg_full_rocket
        inertias_full_rocket[i] = inertia_full_rocket

        x_cog_inertia_stage_1, inertia_stage_1 = stage_1_x_cog_inertia_func(1 - chi)
        d_cp_cg_stage_1 = stage_1_d_cp_cg_func(chi)

        x_cog_inertias_stage_1[i] = x_cog_inertia_stage_1
        d_cp_cgs_stage_1[i] = d_cp_cg_stage_1
        inertias_stage_1[i] = inertia_stage_1

    # Clean to remove NaNs by taking the next valid value
    x_cog_inertias_full_rocket = np.array([x for x in x_cog_inertias_full_rocket if str(x) != 'nan'])
    d_cp_cgs_full_rocket = np.array([x for x in d_cp_cgs_full_rocket if str(x) != 'nan'])
    inertias_full_rocket = np.array([x for x in inertias_full_rocket if str(x) != 'nan'])

    x_cog_inertias_stage_1 = np.array([x for x in x_cog_inertias_stage_1 if str(x) != 'nan'])
    d_cp_cgs_stage_1 = np.array([x for x in d_cp_cgs_stage_1 if str(x) != 'nan'])
    inertias_stage_1 = np.array([x for x in inertias_stage_1 if str(x) != 'nan'])



    # Write to csv, a csv for each array make folder for each rocket stage
    os.makedirs('data/rocket_parameters/lut_creation_rocket_functions', exist_ok=True)
    os.makedirs('data/rocket_parameters/lut_creation_rocket_functions/subrocket_0', exist_ok=True)
    os.makedirs('data/rocket_parameters/lut_creation_rocket_functions/subrocket_2', exist_ok=True)

    # subrocket_0 i.e. full rocket
    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_0/0_x_cog_inertias.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'X cog inertia [m]'])
        for propellant_mass, x_cog_inertia in zip(propellant_masses, x_cog_inertias_full_rocket):
            writer.writerow([propellant_mass, x_cog_inertia])

    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_0/0_d_cp_cgs.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'd_cp_cg [m]'])
        for propellant_mass, d_cp_cg in zip(propellant_masses, d_cp_cgs_full_rocket):
            writer.writerow([propellant_mass, d_cp_cg])

    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_0/0_inertias.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'Inertia [kg m^2]'])
        for propellant_mass, inertia in zip(propellant_masses, inertias_full_rocket):
            writer.writerow([propellant_mass, inertia])

    # subrocket_2 i.e. stage 1
    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_2/2_x_cog_inertias.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'X cog inertia [m]'])
        for propellant_mass, x_cog_inertia in zip(propellant_masses, x_cog_inertias_stage_1):
            writer.writerow([propellant_mass, x_cog_inertia])

    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_2/2_d_cp_cgs.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'd_cp_cg [m]'])
        for propellant_mass, d_cp_cg in zip(propellant_masses, d_cp_cgs_stage_1):
            writer.writerow([propellant_mass, d_cp_cg])

    with open('data/rocket_parameters/lut_creation_rocket_functions/subrocket_2/2_inertias.csv', 'w') as file:
        writer = csv.writer(file)
        # Column names
        writer.writerow(['Propellant mass [kg]', 'Inertia [kg m^2]'])
        for propellant_mass, inertia in zip(propellant_masses, inertias_stage_1):
            writer.writerow([propellant_mass, inertia])
