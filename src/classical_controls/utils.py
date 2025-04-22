def PD_controller_single_step(Kp, Kd, N, error, previous_error, previous_derivative, dt):
    # Proportional term
    P_term = Kp * error
    
    # Derivative term with low-pass filter
    derivative = (error - previous_error) / dt
    D_term = Kd * (N * derivative + (1 - N * dt) * previous_derivative)
    
    # Control action
    control_action = P_term + D_term
    
    return control_action, derivative