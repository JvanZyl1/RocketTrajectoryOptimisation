import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src.envs.rockets_physics import rocket_physics_fcn, compile_physics
from src.envs.disturbance_generator import VKDisturbanceGenerator

class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
    
    def add_result(self, test_name, passed, details=None):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'details': details
        })
    
    def save_to_csv(self, filename):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Passed', 'Details'])
            for result in self.results:
                writer.writerow([
                    result['test_name'],
                    result['passed'],
                    str(result['details']) if result['details'] else ''
                ])

def test_ascent_phase(results):
    """Test rocket physics during ascent phase."""
    test_name = "Ascent Phase"
    try:
        # Initialize parameters from compile_physics
        dt = 0.01
        flight_phase = 'subsonic'
        physics_step = compile_physics(dt, flight_phase)
        
        # Initial state: [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
        state = np.array([
            0.0,  # x
            0.0,  # y
            0.0,  # vx
            0.0,  # vy
            0.0,  # theta
            0.0,  # theta_dot
            0.0,  # gamma
            0.0,  # alpha
            1000.0,  # mass
            500.0,  # mass_propellant
            0.0    # time
        ])
        
        # Actions: [gimbal_angle, throttle]
        actions = np.array([0.0, 0.5])
        
        # Run physics step
        new_state, info = physics_step(state, actions, None)
        
        # Verify results
        assert not np.any(np.isnan(new_state)), "State contains NaN values"
        assert not np.any(np.isnan(info.values())), "Info contains NaN values"
        assert new_state[8] < state[8], "Mass should decrease due to propellant consumption"
        assert new_state[9] < state[9], "Propellant mass should decrease"
        assert new_state[10] > state[10], "Time should increase"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_flip_over_phase(results):
    """Test rocket physics during flip-over phase."""
    test_name = "Flip-over Phase"
    try:
        # Initialize parameters
        dt = 0.01
        flight_phase = 'flip_over_boostbackburn'
        physics_step = compile_physics(dt, flight_phase)
        
        # Initial state
        state = np.array([
            1000.0,  # x
            1000.0,  # y
            100.0,   # vx
            -100.0,  # vy
            math.pi, # theta
            0.0,     # theta_dot
            math.pi, # gamma
            0.0,     # alpha
            1000.0,  # mass
            500.0,   # mass_propellant
            0.0      # time
        ])
        
        # Actions: [gimbal_angle]
        actions = np.array([0.5])
        gimbal_angle_deg_prev = 0.0
        
        # Run physics step
        new_state, info = physics_step(state, actions, gimbal_angle_deg_prev, None)
        
        # Verify results
        assert not np.any(np.isnan(new_state)), "State contains NaN values"
        assert not np.any(np.isnan(info.values())), "Info contains NaN values"
        assert abs(new_state[4] - state[4]) > 0, "Pitch angle should change during flip-over"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_ballistic_arc(results):
    """Test rocket physics during ballistic arc phase."""
    test_name = "Ballistic Arc Phase"
    try:
        # Initialize parameters
        dt = 0.01
        flight_phase = 'ballistic_arc_descent'
        physics_step = compile_physics(dt, flight_phase)
        
        # Initial state
        state = np.array([
            2000.0,  # x
            2000.0,  # y
            0.0,     # vx
            -100.0,  # vy
            math.pi, # theta
            0.0,     # theta_dot
            math.pi, # gamma
            0.0,     # alpha
            1000.0,  # mass
            500.0,   # mass_propellant
            0.0      # time
        ])
        
        # Actions: [RCS_throttle]
        actions = np.array([0.5])
        
        # Run physics step
        new_state, info = physics_step(state, actions, None)
        
        # Verify results
        assert not np.any(np.isnan(new_state)), "State contains NaN values"
        assert not np.any(np.isnan(info.values())), "Info contains NaN values"
        assert new_state[3] < state[3], "Vertical velocity should decrease due to gravity"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))


def test_wind_disturbance(results):
    """Test rocket physics with wind disturbance."""
    test_name = "Wind Disturbance"
    try:
        # Initialize parameters
        dt = 0.01
        flight_phase = 'subsonic'
        physics_step = compile_physics(dt, flight_phase)
        
        # Create wind generator
        wind_generator = VKDisturbanceGenerator()
        
        # Initial state
        state = np.array([
            0.0,  # x
            1000.0,  # y
            0.0,  # vx
            0.0,  # vy
            0.0,  # theta
            0.0,  # theta_dot
            0.0,  # gamma
            0.0,  # alpha
            1000.0,  # mass
            500.0,  # mass_propellant
            0.0    # time
        ])
        
        # Actions: [gimbal_angle, throttle]
        actions = np.array([0.0, 0.5])
        
        # Run physics step with wind
        new_state, info = physics_step(state, actions, wind_generator)
        
        # Verify results
        assert not np.any(np.isnan(new_state)), "State contains NaN values"
        assert not np.any(np.isnan(info.values())), "Info contains NaN values"
        assert 'wind_force' in info, "Wind force should be in info dictionary"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def run_all_tests():
    results = TestResults()
    
    # Run all tests
    test_ascent_phase(results)
    test_flip_over_phase(results)
    test_ballistic_arc(results)
    test_wind_disturbance(results)
    
    # Print results
    print("\n=== Rocket Physics Verification Test Results ===")
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    print("\nDetailed Results:")
    for result in results.results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{status}: {result['test_name']}")
        if not result['passed'] and result['details']:
            print(f"  Details: {result['details']}")
    
    # Save results
    os.makedirs('results/verification/rocket_physics_verification', exist_ok=True)
    results.save_to_csv('results/verification/rocket_physics_verification/test_results.csv')

if __name__ == "__main__":
    run_all_tests() 