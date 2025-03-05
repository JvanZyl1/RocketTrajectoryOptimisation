import numpy as np
from scipy.optimize import fsolve


mu = 398602 * 1e9  # Gravitational parameter [m^3/s^2]
g0 = 9.80665  # Gravity constant on Earth [m/s^2]

def staging_reusable_rocketry(semi_major_axis : float,
                              mass_payload : float,
                              delta_v_loss_ascent : np.array,
                              delta_v_descent : np.array,
                              number_of_stages : int,
                              v_exhaust : np.array,
                              structural_coefficients : np.array,
                              Reusable_indices: np.array,
                              print_bool: bool = False):
    '''
    Based upon the method described by B. Jo and J. Ahn in
    "Optimal staging of reusable launch vehicles considering velocity losses."
    '''
    def kappa_solving(kappa, delta_v_required, 
                 number_of_stages, v_exhaust, structural_coefficients):
            result = -delta_v_required
            for stage in range(number_of_stages):
                term = v_exhaust[stage] * np.log((v_exhaust[stage] - kappa) / (v_exhaust[stage] * structural_coefficients[stage]))
                result += term
                if print_bool:
                    print(f'stage: {stage}, result: {result}')
            return result
         
    
    # Initial guess for kappa
    delta_v_required = np.sqrt(mu / semi_major_axis) + np.sum(delta_v_loss_ascent + delta_v_descent)
    if print_bool:
        print(f'delta_v_required: {delta_v_required}')
    
    kappa_initial = 0.01

    # Define the solver function with all necessary parameters
    def kappa_solver(kappa):
        return kappa_solving(kappa, delta_v_required,
                            number_of_stages, v_exhaust, structural_coefficients)

    # Solve for kappa
    kappa = fsolve(kappa_solver, kappa_initial)[0]
    if print_bool:
        print(f'kappa: {kappa}')

    # Calculate structural coefficients of descent
    structural_coefficients_descent = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            structural_coefficients_descent[stage] = 1/np.exp(delta_v_descent[stage]/v_exhaust[stage])
    if print_bool:
        print(f'structural_coefficients_descent: {structural_coefficients_descent}')

    # Calculate structural coefficients of ascent
    structural_coefficients_ascent = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            structural_coefficients_ascent[stage] = structural_coefficients[stage]/structural_coefficients_descent[stage]
        else:
            structural_coefficients_ascent[stage] = structural_coefficients[stage]
    if print_bool:        
        print(f'structural_coefficients_ascent: {structural_coefficients_ascent}')

    # Compute optimal loss-free velocity increments of each stage
    payload_ratios_opt = np.zeros(number_of_stages)
    delta_vs_ascent_opt = np.zeros(number_of_stages)
    delta_vs_opt = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            payload_ratios_opt[stage] = kappa * structural_coefficients_ascent[stage] \
                                        / ((1 - structural_coefficients_ascent[stage]) * v_exhaust[stage] - kappa)
            delta_vs_ascent_opt[stage] = v_exhaust[stage] * np.log((1 + payload_ratios_opt[stage])/(structural_coefficients_ascent[stage] + payload_ratios_opt[stage]))
        else:
            payload_ratios_opt[stage] = kappa * structural_coefficients[stage] \
                                        / ((1 - structural_coefficients[stage]) * v_exhaust[stage] - kappa)
            delta_vs_opt[stage] = v_exhaust[stage] * np.log((1 + payload_ratios_opt[stage])/(structural_coefficients[stage] + payload_ratios_opt[stage]))
    if print_bool:
        print(f'payload_ratios_opt: {payload_ratios_opt}')
        print(f'delta_vs_ascent_opt: {delta_vs_ascent_opt}')
        print(f'delta_vs_opt: {delta_vs_opt}')

    # Calculate the structural fractions with losses for ascent and descent phases
    structural_coefficients_withlosses_descent = np.zeros(number_of_stages)
    structural_coefficients_withlosses_ascent = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            structural_coefficients_withlosses_descent[stage] = np.exp(-(delta_v_descent[stage])/v_exhaust[stage])
            structural_coefficients_withlosses_ascent[stage] = structural_coefficients[stage]/structural_coefficients_withlosses_descent[stage]
    if print_bool:
        print(f'structural_coefficients_withlosses_descent: {structural_coefficients_withlosses_descent}')
        print(f'structural_coefficients_withlosses_ascent: {structural_coefficients_withlosses_ascent}')

    # Payload ratios with losses
    delta_vs_ascent = np.zeros(number_of_stages)
    payload_ratio_withlosses_opt = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            delta_vs_ascent[stage] = delta_vs_ascent_opt[stage] + delta_v_loss_ascent[stage]
            payload_ratio_withlosses_opt[stage] = (structural_coefficients_withlosses_ascent[stage] * np.exp(delta_vs_ascent[stage]/v_exhaust[stage]) - 1) \
                                                            / (1 - np.exp(delta_vs_ascent[stage]/v_exhaust[stage]))
        else:
            delta_vs_opt[stage] = delta_vs_opt[stage] + delta_v_loss_ascent[stage]
            payload_ratio_withlosses_opt[stage] = (structural_coefficients[stage] * np.exp(delta_vs_opt[stage]/v_exhaust[stage]) - 1) \
                                                            / (1 - np.exp(delta_vs_opt[stage]/v_exhaust[stage]))
    if print_bool:
        print(f'payload_ratio_withlosses_opt: {payload_ratio_withlosses_opt}')
        print(f'delta_vs_ascent: {delta_vs_ascent}')
        print(f'delta_vs_opt: {delta_vs_opt}')

    m_L_losses = np.zeros(number_of_stages+1) # To include payload
    m_L_losses[-1] = mass_payload
    # m_L_losses[i-1] = (1/lambda_losses[i] + 1) * m_L_losses[i]
    # reverse order
    for stage in range(number_of_stages, 0, -1):
        m_L_losses[stage-1] = (1/payload_ratio_withlosses_opt[stage-1] + 1) * m_L_losses[stage]
    if print_bool:
        print(f'm_L_losses: {m_L_losses}')

    # Structural masses and propellant masses for ascent
    structural_masses_withlosses_ascent = np.zeros(number_of_stages)
    propellant_masses_withlosses_ascent = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            structural_masses_withlosses_ascent[stage] = (structural_coefficients_withlosses_ascent[stage] / payload_ratio_withlosses_opt[stage]) * m_L_losses[stage]
            propellant_masses_withlosses_ascent[stage] = ((1 - structural_coefficients_withlosses_ascent[stage]) / payload_ratio_withlosses_opt[stage]) * m_L_losses[stage]
        else:
            structural_masses_withlosses_ascent[stage] = (structural_coefficients[stage] / payload_ratio_withlosses_opt[stage]) * m_L_losses[stage]
            propellant_masses_withlosses_ascent[stage] = ((1 - structural_coefficients[stage]) / payload_ratio_withlosses_opt[stage]) * m_L_losses[stage]
    if print_bool:
        print(f'structural_masses_withlosses_ascent: {structural_masses_withlosses_ascent}')
        print(f'propellant_masses_withlosses_ascent: {propellant_masses_withlosses_ascent}')

    # Structural Mass Ascent of reusable stages = Structural Mass Descent + Descent Propellant Mass
    structural_masses_withlosses_descent = np.zeros(number_of_stages)
    propellant_masses_withlosses_descent = np.zeros(number_of_stages)
    for stage in range(number_of_stages):
        if stage in Reusable_indices:
            structural_masses_withlosses_descent[stage] = structural_masses_withlosses_ascent[stage] * structural_coefficients_withlosses_descent[stage]
            propellant_masses_withlosses_descent[stage] = structural_masses_withlosses_ascent[stage] - structural_masses_withlosses_descent[stage]
            assert structural_masses_withlosses_ascent[stage] == (structural_masses_withlosses_descent[stage] + propellant_masses_withlosses_descent[stage])
    if print_bool:
        print(f'structural_masses_withlosses_descent: {structural_masses_withlosses_descent}')
        print(f'propellant_masses_withlosses_descent: {propellant_masses_withlosses_descent}')
    
    '''
    Return
    m_{s_a,i} : structural mass of stage i for ascent phase
    m_{s_d,i} : structural mass of stage i for descent phase
    m_{p_a,i} : propellant mass of stage i for ascent phase
    m_{p_d,i} : propellant mass of stage i for descent phase
    '''
    structural_masses_ascent = structural_masses_withlosses_ascent
    propellant_masses_ascent = propellant_masses_withlosses_ascent
    structural_masses_descent = structural_masses_withlosses_descent
    propellant_masses_descent = propellant_masses_withlosses_descent

    initial_mass = np.sum(structural_masses_ascent + propellant_masses_ascent) + mass_payload
    mass_at_stage_1_ascent_burnout = initial_mass - propellant_masses_ascent[0]
    mass_of_rocket_at_stage_1_separation = mass_at_stage_1_ascent_burnout - structural_masses_ascent[0]
    mass_of_stage_1_at_separation = structural_masses_ascent[0] # := Structural Mass + Propellant Mass of stage 1 descent
    mass_at_stage_1_descent_burnout = mass_of_stage_1_at_separation - propellant_masses_descent[0]
    mass_at_stage_2_ascent_burnout = mass_of_rocket_at_stage_1_separation - propellant_masses_ascent[1]
    mass_at_stage_2_separation = mass_payload
    stage_masses_dict = {
        'structural_mass_stage_1_ascent': structural_masses_ascent[0],
        'propellant_mass_stage_1_ascent': propellant_masses_ascent[0],
        'structural_mass_stage_1_descent': structural_masses_descent[0],
        'propellant_mass_stage_1_descent': propellant_masses_descent[0],
        'structural_mass_stage_2_ascent': structural_masses_ascent[1],
        'propellant_mass_stage_2_ascent': propellant_masses_ascent[1],
        'initial_mass': initial_mass,
        'mass_at_stage_1_ascent_burnout': mass_at_stage_1_ascent_burnout,
        'mass_of_rocket_at_stage_1_separation': mass_of_rocket_at_stage_1_separation,
        'mass_of_stage_1_at_separation': mass_of_stage_1_at_separation,
        'mass_at_stage_1_descent_burnout': mass_at_stage_1_descent_burnout,
        'mass_at_stage_2_ascent_burnout': mass_at_stage_2_ascent_burnout,
        'mass_at_stage_2_separation': mass_at_stage_2_separation
    }

    return stage_masses_dict