def PD_controller_single_step(Kp, Kd, N, error, previous_error, previous_derivative, dt):
    # Proportional term
    P_term = Kp * error
    
    # Derivative term with low-pass filter
    derivative = (error - previous_error) / dt
    D_term = Kd * (N * derivative + (1 - N * dt) * previous_derivative)
    
    # Control action
    control_action = P_term + D_term
    
    return control_action, derivative

class PIDController:
    def __init__(self, Kp, Ki, Kd, N, dt, output_limits=None, previous_error=0.0, previous_derivative=0.0, initial_integral=0.0):
        """
        PID controller with derivative low-pass filter
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            N: Filter coefficient for derivative term
            dt: Time step
            output_limits: Tuple of (min, max) output limits
            previous_error: Initial previous error value
            previous_derivative: Initial previous derivative value
            initial_integral: Initial integral value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N
        self.dt = dt
        self.output_limits = output_limits
        
        # Initialize internal state with provided values
        self.previous_error = previous_error
        self.previous_derivative = previous_derivative
        self.integral = initial_integral
    
    def reset(self, previous_error=0.0, previous_derivative=0.0, initial_integral=0.0):
        """
        Reset the controller state
        
        Args:
            previous_error: Reset previous error to this value
            previous_derivative: Reset previous derivative to this value
            initial_integral: Reset integral to this value
        """
        self.previous_error = previous_error
        self.previous_derivative = previous_derivative
        self.integral = initial_integral
    
    def step(self, error):
        """
        Compute one step of the PID control
        
        Args:
            error: Current error value
            
        Returns:
            control_action: The control output
        """
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        I_term = self.Ki * self.integral
        
        # Derivative term with low-pass filter
        derivative = (error - self.previous_error) / self.dt
        D_term = self.Kd * (self.N * derivative + (1 - self.N * self.dt) * self.previous_derivative)
        
        # Control action
        control_action = P_term + I_term + D_term
        
        # Apply output limits if specified
        if self.output_limits is not None:
            control_action = max(min(control_action, self.output_limits[1]), self.output_limits[0])
        
        # Update state for next step
        self.previous_error = error
        self.previous_derivative = derivative
        
        return control_action