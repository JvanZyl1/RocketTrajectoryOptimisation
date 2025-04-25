from scipy.optimize import minimize
from src.classical_controls.re_entry_burn import ReEntryBurn
import numpy as np


def objective_func_lambda(individual):
    re_entry_burn = ReEntryBurn(individual)
    re_entry_burn.run_closed_loop()
    obj = -re_entry_burn.performance_metrics()
    re_entry_burn.plot_results()
    return obj

def constraint_func_lambda(individual):
    re_entry_burn = ReEntryBurn(individual)
    re_entry_burn.run_closed_loop()
    max_pressure = max(re_entry_burn.dynamic_pressure_vals)
    cons = re_entry_burn.Q_max - max_pressure
    return cons

if __name__ == '__main__':
    # Kp_mach, Kp_pitch, Kd_pitch, N_pitch
    bounds = [(-1, 1), (-1, 1), (-1, 1)]
    x0 = [0.05, 0.3, -0.1]
    
    print("Starting optimization with trust-constr method...")
    result = minimize(
        objective_func_lambda,
        method='trust-constr',
        bounds=bounds,
        constraints = [{'type': 'ineq', 'fun': constraint_func_lambda}],
        x0 = x0,                  
        options={
            'maxiter': 50,
            'verbose': 2,
            'gtol': 1e-6,    # Tolerance for termination by the norm of the gradient
            'xtol': 1e-6,    # Tolerance for termination by the change of the independent variable
            'barrier_tol': 1e-8,  # Stricter barrier tolerance to enforce constraints
            'initial_tr_radius': 1.0  # Initial trust radius
        }
    )