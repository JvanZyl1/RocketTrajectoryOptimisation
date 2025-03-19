import numpy as np
import matplotlib.pyplot as plt

class PIDcontroller:
    def __init__(self, kp, ki, kd, delta_t = 0.004):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_integral = 0
        self.previous_error = 0
        self.delta_t = delta_t

    def update(self, error):
        self.error_integral += error * self.delta_t
        self.previous_error = error
        error_derivative = (error - self.previous_error) / self.delta_t

        p = self.kp * error
        i = self.ki * self.error_integral
        d = self.kd * error_derivative

        return p + i + d
    
    def reset(self):
        self.error_integral = 0
        self.previous_error = 0

class RateLimiter:
    def __init__(self, ratelimiter, saturation_lower, saturation_upper, initial_reference_t = 0):
        self.ratelimiter = ratelimiter
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper

        self.reference_t_1 = initial_reference_t
        self.initial_reference_t = initial_reference_t

    def rate_limit(self, reference_t):
        delta_setpoint = reference_t - self.reference_t_1
        if delta_setpoint > self.ratelimiter:
            reference_t = self.reference_t_1 + self.ratelimiter
        elif delta_setpoint < -self.ratelimiter:
            reference_t = self.reference_t_1 - self.ratelimiter
        
        reference_t = np.clip(reference_t, self.saturation_lower, self.saturation_upper)

        self.reference_t_1 = reference_t
        return reference_t
    
    def reset(self):
        self.reference_t_1 = self.initial_reference_t

class RL_PID_RL:
    # Input RL + PID + Output RL
    def __init__(self, params, delta_t = 0.004):
        self.kp = params['kp']
        self.ki = params['ki']
        self.kd = params['kd']

        self.delta_t = delta_t

        self.reference_RL = params['reference_RL']
        self.reference_saturation_lower = params['reference_saturation_lower']
        self.reference_saturation_upper = params['reference_saturation_upper']
        self.initial_reference_t = params['initial_reference_t']

        self.controller_RL = params['controller_RL']
        self.controller_saturation_lower = params['controller_saturation_lower']
        self.controller_saturation_upper = params['controller_saturation_upper']

        # 1) Input rate limiter and saturation
        self.rate_limiter_input = RateLimiter(self.reference_RL,
                                              self.reference_saturation_lower,
                                              self.reference_saturation_upper,
                                              self.initial_reference_t)

        # 2) PID controller
        self.pid_controller = PIDcontroller(self.kp,
                                            self.ki,
                                            self.kd,
                                            self.delta_t)
        
        # 3) Output rate limiter and saturation
        self.rate_limiter_output = RateLimiter(self.controller_RL,
                                               self.controller_saturation_lower,
                                               self.controller_saturation_upper)
        
        self.reference_t_1 = self.initial_reference_t
        #self.controller_output_t_1 = 0
        self.controlleroutput_t_1 = 0
        
    def update(self, reference):
        # 1) Input rate limiter and saturation
        reference_sat = self.rate_limiter_input.rate_limit(reference)
        
        # 2) PID controller
        error = reference_sat - self.reference_t_1
        controller_output_presat_t = self.pid_controller.update(error)
        
        # 3) Output rate limiter and saturation
        controller_output_t = self.rate_limiter_output.rate_limit(controller_output_presat_t)

        # 4) Update reference_t_1 and controlleroutput_t_1
        self.reference_t_1 = reference_sat
        self.controlleroutput_t_1 = controller_output_t

        
        return controller_output_t, reference_sat, controller_output_presat_t
    
    def reset(self):
        self.rate_limiter_input.reset()
        self.pid_controller.reset()
        self.rate_limiter_output.reset()
        self.reference_t_1 = self.initial_reference_t
        self.controlleroutput_t_1 = 0

class ClosedLoopFB:
    def __init__(self, controller_params, model, delta_t):
        self.controller = RL_PID_RL(controller_params, delta_t)
        self.controller_output_model = model
        self.reference = 0
        self.output = 0
        self.controller_output = 0
        self.reference_saturated = 0
        self.controller_output_presaturated = 0
        self.reference_array = []
        self.output_array = []
        self.controller_output_array = []
        self.reference_saturated_array = []
        self.controller_output_presaturated_array = []
        self.delta_t = delta_t

        self.output_array = []

    def update(self, reference):
        self.reference = reference
        controller_output, reference_sat, controller_output_presat = self.controller.update(reference)
        output = self.controller_output_model.update(controller_output)
        self.output = output
        self.controller_output = controller_output
        self.reference_saturated = reference_sat
        self.controller_output_presaturated = controller_output_presat
        return output
    
    def update_array(self, reference_array):
        self.reference_array = reference_array
        output_array = []
        controller_output_array = []
        reference_saturated_array = []
        controller_output_presaturated_array = []
        for reference in self.reference_array:
            output = self.update(reference)
            output_array.append(output)
            controller_output_array.append(self.controller_output)
            reference_saturated_array.append(self.reference_saturated)
            controller_output_presaturated_array.append(self.controller_output_presaturated)
            
        self.output_array = output_array
        self.controller_output_array = controller_output_array
        self.reference_saturated_array = reference_saturated_array
        self.controller_output_presaturated_array = controller_output_presaturated_array
        return output_array
    
    def plot_results_generic(self):
        # Plot subplot: 1: (reference & accleration) 2: controller_output
        time_array = np.arange(0, self.delta_t*len(self.reference_array), self.delta_t)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(time_array, self.reference_array, label='Reference')
        ax[0].plot(time_array, self.output_array, label='Output')
        ax[0].plot(time_array, self.reference_saturated_array, '--', label='Reference Saturated')
        ax[0].legend()
        ax[0].set_xlabel('Time (s)')  # Fixed
        ax[0].set_ylabel('Output (-)')  # Fixed and corrected unit
        ax[0].set_title('Reference & Output')

        ax[1].plot(time_array, self.controller_output_array, label='Controller Output')
        ax[1].plot(time_array, self.controller_output_presaturated_array, '--', label='Controller Output Presaturated')
        ax[1].legend()
        ax[1].set_xlabel('Time (s)')  # Fixed
        ax[1].set_ylabel('Controller Output (-)')  # Assuming controller_output is in percentage
        ax[1].set_title('Controller Output')
        plt.close()

    def update_gains(self, kp, ki, kd):
        # For ease of use
        self.controller.pid_controller.kp = kp
        self.controller.pid_controller.ki = ki
        self.controller.pid_controller.kd = kd
        self.reset()
        return None
    
    def individual_update_model(self, individual):
        kp = individual[0]
        ki = individual[1]
        kd = individual[2]
        self.update_gains(kp, ki, kd)
    
    def calculate_error(self, measured_output, predicted_output):
        error = 0
        for i in range(len(measured_output)):
            error += (measured_output[i] - predicted_output[i])**2
        return error

    def objective_function(self, individual, input, measured_output):
        self.individual_update_model(individual)
        predicted_output = self.update_array(input)
        error = self.calculate_error(measured_output, predicted_output)
        return error
    
    def reset(self):
        self.controller.reset()
        self.controller_output_model.reset()
        self.reference = 0
        self.output = 0
        self.controller_output = 0
        self.reference_saturated = 0
        self.controller_output_presaturated = 0
        self.reference_array = []
        self.output_array = []
        self.controller_output_array = []
        self.reference_saturated_array = []
        self.controller_output_presaturated_array = []
        self.output = []
