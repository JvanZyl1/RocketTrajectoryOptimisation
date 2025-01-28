import numpy as np
from scipy.optimize import fsolve

def staging_expendable(number_of_stages,
            specific_impulses_vacuum,
            structural_coefficients,
            payload_mass,
            semi_major_axis,
            delta_v_loss=2000,
            mu=398602 * 1e9,
            g0=9.81):
    """
    Provides the masses for the different stages.

    Parameters:
        number_of_stages: int - Number of stages
        specific_impulses_vacuum: list - Specific impulse for each stage [s]
        structural_coefficients: list - Structural coefficient for each stage
        payload_mass: float - Payload mass [kg]
        semi_major_axis: float - Semi-major axis [m]
        delta_v_loss: float - Delta-V for losses [m/s]
    
    Returns:
        Tuple containing:
        - Initial mass [kg]
        - Subrocket masses [kg]
        - Stage masses [kg]
        - Structural masses [kg]
        - Propellant masses [kg]
        - Delta-V required [m/s]
        - Stage Delta-V [m/s]
        - Payload ratio (lambda) for each stage
        - Mass ratio for each stage
    """
    # Delta_V required [m/s]
    delta_v_required = np.sqrt(mu /semi_major_axis) + delta_v_loss

    # Exhaust velocity [m/s]
    v_exhaust = np.array(specific_impulses_vacuum) * g0  
    v_exhaust_mean = np.mean(v_exhaust)  # Mean characteristic velocity [m/s]
    structural_coefficients_mean = np.mean(structural_coefficients)  # Mean structural coefficient

    # Initial Lagrange Multiplier
    non_dimensional_delta_v = delta_v_required / v_exhaust_mean
    total_payload_ratio = ((np.exp(-non_dimensional_delta_v / number_of_stages) -\
                    structural_coefficients_mean) / (1 - structural_coefficients_mean))**number_of_stages
    stage_payload_ratio = total_payload_ratio**(1 / number_of_stages)
    stage_mass_ratio = 1 / (structural_coefficients_mean * (1 - stage_payload_ratio) + stage_payload_ratio)
    initial_lagrange_multiplier = 1 / (stage_mass_ratio * v_exhaust_mean * structural_coefficients_mean - v_exhaust_mean)

    # Finding Lagrange Multiplier
    def func(lagrange_multiplier):
        result = -delta_v_required
        for i in range(number_of_stages):
            result += v_exhaust[i] * np.log((1 + lagrange_multiplier * v_exhaust[i]) / (lagrange_multiplier * v_exhaust[i] \
                                                                                        * structural_coefficients[i]))
        return result

    lagrange_multiplier = fsolve(func, initial_lagrange_multiplier)[0]

    mass_ratios = np.zeros(number_of_stages)
    payload_ratios = np.zeros(number_of_stages)
    for i in range(number_of_stages - 1, -1, -1):
        mass_ratios[i] = (1 + lagrange_multiplier * v_exhaust[i]) / (lagrange_multiplier * v_exhaust[i] * structural_coefficients[i])
        payload_ratios[i] = (1 - structural_coefficients[i] * mass_ratios[i]) / ((1 - structural_coefficients[i]) * mass_ratios[i])

    lambda_total = np.prod(payload_ratios)
    delta_v_required_stages = -v_exhaust * np.log(structural_coefficients * (1 - payload_ratios) + payload_ratios)

    # Error checking
    if abs(np.sum(delta_v_required_stages) - delta_v_required) > 1e-5:
        raise ValueError("The staging is not correct, delta_v_required_stages is not equal to delta_v_required")

    # Calculate the second derivative check for stability of staging parameters
    second_derivative_check = -(1 + lagrange_multiplier * v_exhaust) / (mass_ratios**2) + \
                            (structural_coefficients / (1 - structural_coefficients * mass_ratios))**2

    # Ensure all stages pass the second derivative condition for a stable solution
    if np.any(second_derivative_check < 0):
        raise ValueError("A local maximum has been reached, or stages incorrectly defined")

    # Output masses
    initial_mass = payload_mass / lambda_total  # Initial mass [kg]
    sub_stage_masses = np.zeros(number_of_stages + 1)
    sub_stage_masses[-1] = payload_mass

    stage_masses = np.zeros(number_of_stages)
    structural_masses = np.zeros(number_of_stages)
    propellant_masses = np.zeros(number_of_stages)

    for i in range(number_of_stages - 1, -1, -1):
        sub_stage_masses[i] = sub_stage_masses[i + 1] / payload_ratios[i]       # Subrocket mass [kg]
        stage_masses[i] = sub_stage_masses[i] - sub_stage_masses[i + 1]         # Stage mass [kg]
        structural_masses[i] = stage_masses[i] * structural_coefficients[i]     # Structural mass [kg]
        propellant_masses[i] = stage_masses[i] - structural_masses[i]           # Propellant mass [kg]

    return initial_mass, sub_stage_masses, stage_masses, structural_masses, propellant_masses, \
        delta_v_required, delta_v_required_stages, payload_ratios, mass_ratios

def compute_stage_properties(initial_mass,
                             stage_masses,
                             propellant_masses,
                             structural_masses,
                             specific_impulses_vacuum,
                             g0,
                             max_accelerations,
                             mass_fairing=50):
    """
    Computes properties for each stage of a multistage rocket.

    Parameters:
        initial_mass (float): Initial mass [kg]
        stage_masses (list): Stage masses [kg]
        propellant_masses (list): Propellant masses for each stage [kg]
        structural_masses (list): Structural masses for each stage [kg]
        specific_impulses_vacuum (list): Specific impulse for each stage [s]
        g0 (float): Standard gravity [m/s^2]
        max_accelerations (list): Maximum acceleration for each stage [m/s^2]
        mass_fairing (float, optional): Fairing mass [kg]. Defaults to 50.
    
    Returns:
        dict: A dictionary containing updated structural masses, thrusts, 
              mass flow rates, and burn times for each stage.
    """
    num_stages = len(stage_masses)

    # Adjust structural masses to account for the fairing
    structural_masses[0] += mass_fairing
    structural_masses[-1] -= mass_fairing

    # Initialize lists to store results
    burn_out_masses = []
    separation_masses = []
    thrusts = []
    mass_flow_rates = []
    burn_times = []

    # Effective exhaust velocities
    v_exhaust = np.array(specific_impulses_vacuum) * g0

    # Iteratively compute properties for each stage
    current_mass = initial_mass
    for i in range(num_stages):
        # Mass at burn out
        burn_out_mass = current_mass - propellant_masses[i]
        burn_out_masses.append(burn_out_mass)

        # Mass at stage separation (current stage's initial mass)
        separation_mass = current_mass - stage_masses[i]
        separation_masses.append(separation_mass)

        # Thrust for the stage
        thrust = burn_out_mass * max_accelerations[i]
        thrusts.append(thrust)

        # Mass flow rate for the stage
        mass_flow_rate = thrust / v_exhaust[i]
        mass_flow_rates.append(mass_flow_rate)

        # Burn time for the stage
        burn_time = propellant_masses[i] / mass_flow_rate
        burn_times.append(burn_time)

        # Update the current mass for the next stage
        current_mass = separation_mass

    # Return results as a dictionary
    return {
        "adjusted_structural_masses": structural_masses,
        "burn_out_masses": burn_out_masses,
        "separation_masses": separation_masses,
        "thrusts": thrusts,
        "mass_flow_rates": mass_flow_rates,
        "burn_times": burn_times,
    }
