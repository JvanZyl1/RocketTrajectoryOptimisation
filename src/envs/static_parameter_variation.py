import dill

from src.envs.utils.aerodynamic_coefficients import rocket_CL, rocket_CD

class LiftCoefficient:
    def __init__(self):
        self.CL_func_lambda = lambda alpha, M: rocket_CL(alpha, M)

    def __call__(self, alpha, M):
        pass

    def reset(self):
        pass

class DragCoefficient:
    def __init__(self):
        self.CD_func_lambda = lambda M: rocket_CD(M)

    def __call__(self, alpha, M):
        pass

    def reset(self):
        pass

class CenterOfPressure:
    def __init__(self,
                 flight_phase : str):
        assert flight_phase in ['subsonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']

        with open('data/rocket_parameters/rocket_functions.pkl', 'rb') as f:  
            rocket_functions = dill.load(f)

        if flight_phase in ['subsonic', 'supersonic']:
            self.cop_func_lambda = lambda alpha, M: rocket_functions['cop_subrocket_0_lambda'](alpha, M, x_cop_alpha_subsonic, x_cop_alpha_supersonic, x_cop_machsupersonic)
        elif flight_phase in ['flip_over_boostbackburn', 'ballistic_arc_descent', 're_entry_burn']:
            self.cop_func_lambda = lambda alpha, M: rocket_functions['cop_subrocket_2_lambda'](alpha, M, x_cop_alpha_subsonic, x_cop_alpha_supersonic, x_cop_machsupersonic)
        # Exo ascent : cop_func_full_rocket_descent = lambda alpha, M: rocket_functions['cop_subrocket_1_lambda'](alpha, M, x_cop_alpha_subsonic, x_cop_alpha_supersonic, x_cop_machsupersonic)

    def __call__(self, alpha, M):
        return self.cop_func_lambda(alpha, M)
    
    def reset(self):
        pass