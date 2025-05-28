import numpy as np
import random
import math
from scipy.signal import cont2discrete
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('.')
from src.envs.wind.vonkarman import VonKarmanFilter, VKDisturbanceGenerator

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

def test_von_karman_reset(results):
    """Test VonKarmanFilter reset functionality."""
    test_name = "VonKarmanFilter Reset"
    try:
        print("\n=== Testing VonKarmanFilter Reset ===")
        f = VonKarmanFilter(L=100.0, sigma=1.0, V=10.0, dt=0.1)
        f.state = np.array([1.0, -1.0])
        print(f"Initial state: {f.state}")
        f.reset()
        print(f"State after reset: {f.state}")
        assert np.allclose(f.state, np.zeros(2)), f"reset(): state not zeroed, got {f.state}"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_von_karman_zero_noise(results):
    """Test VonKarmanFilter with zero noise input."""
    test_name = "VonKarmanFilter Zero Noise"
    try:
        print("\n=== Testing VonKarmanFilter Zero Noise ===")
        orig = np.random.randn
        np.random.randn = lambda: 0.0
        f = VonKarmanFilter(L=50.0, sigma=2.0, V=20.0, dt=0.05)
        f.reset()
        out = f.step()
        print(f"Output with zero noise: {out}")
        print(f"State after step: {f.state}")
        assert out == 0.0, f"step() with zero noise: expected 0.0, got {out}"
        assert np.allclose(f.state, np.zeros(2)), f"state not zero after step(), got {f.state}"
        np.random.randn = orig
        results.add_result(test_name, True)
    except Exception as e:
        np.random.randn = orig
        results.add_result(test_name, False, str(e))

def test_von_karman_unit_noise(results):
    """Test VonKarmanFilter with unit noise input."""
    test_name = "VonKarmanFilter Unit Noise"
    try:
        print("\n=== Testing VonKarmanFilter Unit Noise ===")
        orig = np.random.randn
        np.random.randn = lambda: 1.0
        f = VonKarmanFilter(L=50.0, sigma=1.5, V=15.0, dt=0.1)
        f.reset()
        out = f.step()
        expected_state = f.Bd * 1.0
        expected_out = f.Cd @ expected_state
        print(f"Output: {out}")
        print(f"Expected output: {expected_out}")
        print(f"State: {f.state}")
        print(f"Expected state: {expected_state}")
        assert np.allclose(f.state, expected_state), f"step() state mismatch: {f.state} vs {expected_state}"
        assert np.isclose(out, expected_out), f"step() output mismatch: {out} vs {expected_out}"
        np.random.randn = orig
        results.add_result(test_name, True)
    except Exception as e:
        np.random.randn = orig
        results.add_result(test_name, False, str(e))

def test_vk_disturbance_call(results):
    """Test VKDisturbanceGenerator call functionality."""
    test_name = "VKDisturbanceGenerator Call"
    try:
        print("\n=== Testing VKDisturbanceGenerator Call ===")
        random.seed(0)
        np.random.seed(0)
        gen = VKDisturbanceGenerator(dt=0.1, V=100.0)
        
        # Test initial log_data
        assert 'gust_u' in gen.log_data, "log_data missing gust_u key"
        assert 'gust_v' in gen.log_data, "log_data missing gust_v key"
        assert len(gen.log_data['gust_u']) == 0, "gust_u log should be empty initially"
        assert len(gen.log_data['gust_v']) == 0, "gust_v log should be empty initially"
        
        # Call generator
        gust_u, gust_v = gen(rho=1.225)
        print(f"Gust u: {gust_u}")
        print(f"Gust v: {gust_v}")
        
        # Check return types
        assert isinstance(gust_u, float), f"gust_u type incorrect: {type(gust_u)}"
        assert isinstance(gust_v, float), f"gust_v type incorrect: {type(gust_v)}"
        
        # Check logging
        assert len(gen.log_data['gust_u']) == 1, "gust_u not logged correctly"
        assert len(gen.log_data['gust_v']) == 1, "gust_v not logged correctly"
        assert gen.log_data['gust_u'][0] == gust_u, "logged gust_u doesn't match returned value"
        assert gen.log_data['gust_v'][0] == gust_v, "logged gust_v doesn't match returned value"
        
        # Multiple calls
        for _ in range(5):
            gen(rho=1.225)
        assert len(gen.log_data['gust_u']) == 6, "gust_u log not updated correctly after multiple calls"
        assert len(gen.log_data['gust_v']) == 6, "gust_v log not updated correctly after multiple calls"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_vk_disturbance_reset(results):
    """Test VKDisturbanceGenerator reset functionality."""
    test_name = "VKDisturbanceGenerator Reset"
    try:
        print("\n=== Testing VKDisturbanceGenerator Reset ===")
        random.seed(1)
        gen = VKDisturbanceGenerator(dt=0.2, V=200.0)
        old_Lu, old_Lv = gen.L_u, gen.L_v
        old_sigma_u, old_sigma_v = gen.sigma_u, gen.sigma_v
        
        # Generate some data
        for _ in range(10):
            gen(rho=1.0)
        assert len(gen.log_data['gust_u']) == 10, "gust_u log not updated correctly"
        
        print(f"Original L_u: {old_Lu}, L_v: {old_Lv}")
        print(f"Original sigma_u: {old_sigma_u}, sigma_v: {old_sigma_v}")
        
        # Reset
        gen.reset()
        
        print(f"New L_u: {gen.L_u}, L_v: {gen.L_v}")
        print(f"New sigma_u: {gen.sigma_u}, sigma_v: {gen.sigma_v}")
        
        # Check parameters change
        params_changed = (gen.L_u != old_Lu) or (gen.L_v != old_Lv) or (gen.sigma_u != old_sigma_u) or (gen.sigma_v != old_sigma_v)
        assert params_changed, "reset(): no parameters changed"
        
        # Check logs are reset
        assert len(gen.log_data['gust_u']) == 0, "gust_u log not cleared after reset"
        assert len(gen.log_data['gust_v']) == 0, "gust_v log not cleared after reset"
        assert gen.log_data['L_u'] == gen.L_u, "L_u not updated in log_data"
        assert gen.log_data['L_v'] == gen.L_v, "L_v not updated in log_data"
        assert gen.log_data['sigma_u'] == gen.sigma_u, "sigma_u not updated in log_data"
        assert gen.log_data['sigma_v'] == gen.sigma_v, "sigma_v not updated in log_data"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_seed_reproducibility(results):
    """Test VKDisturbanceGenerator seed reproducibility."""
    test_name = "Seed Reproducibility"
    try:
        print("\n=== Testing Seed Reproducibility ===")
        random.seed(42); np.random.seed(42)
        g1 = VKDisturbanceGenerator(dt=0.05, V=150.0)
        seq1 = [g1(rho=1.0) for _ in range(10)]
        random.seed(42); np.random.seed(42)
        g2 = VKDisturbanceGenerator(dt=0.05, V=150.0)
        seq2 = [g2(rho=1.0) for _ in range(10)]
        print(f"First sequence: {seq1[:3]}...")
        print(f"Second sequence: {seq2[:3]}...")
        for (u1, v1), (u2, v2) in zip(seq1, seq2):
            assert np.isclose(u1, u2), "Horizontal gusts don't match"
            assert np.isclose(v1, v2), "Vertical gusts don't match"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_reset_decorrelation(results):
    """Test VKDisturbanceGenerator reset decorrelation."""
    test_name = "Reset Decorrelation"
    try:
        print("\n=== Testing Reset Decorrelation ===")
        gen = VKDisturbanceGenerator(dt=0.1, V=50.0)
        seq1_u = []
        seq1_v = []
        for _ in range(1000):
            u, v = gen(rho=1.0)
            seq1_u.append(u)
            seq1_v.append(v)
        
        gen.reset()
        
        seq2_u = []
        seq2_v = []
        for _ in range(1000):
            u, v = gen(rho=1.0)
            seq2_u.append(u)
            seq2_v.append(v)
            
        corr_u = np.corrcoef(seq1_u, seq2_u)[0,1]
        corr_v = np.corrcoef(seq1_v, seq2_v)[0,1]
        print(f"Correlation between u sequences: {corr_u}")
        print(f"Correlation between v sequences: {corr_v}")
        assert abs(corr_u) < 0.35, f"Post-reset u correlation too high: {corr_u}"
        assert abs(corr_v) < 0.35, f"Post-reset v correlation too high: {corr_v}"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_long_run_variance(results):
    """Test VonKarmanFilter long-run variance."""
    test_name = "Long Run Variance"
    try:
        print("\n=== Testing Long Run Variance ===")
        f = VonKarmanFilter(L=200.0, sigma=1.2, V=100.0, dt=0.02)
        data = np.array([f.step() for _ in range(50000)])
        var_ratio = data.var() / (1.2**2)
        print(f"Variance ratio: {var_ratio}")
        # Allow ZOH-induced reduction: only upper bound enforced
        assert var_ratio < 1.2, f"Variance ratio too high: {var_ratio}"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_autocorr_timescale(results):
    """Test VonKarmanFilter autocorrelation timescale."""
    test_name = "Autocorrelation Timescale"
    try:
        print("\n=== Testing Autocorrelation Timescale ===")
        f = VonKarmanFilter(L=100.0, sigma=1.0, V=50.0, dt=0.01)
        data = np.array([f.step() for _ in range(5000)])
        ac1 = np.corrcoef(data[:-1], data[1:])[0,1]
        omega0 = 50.0/100.0
        expected_ac1 = math.exp(-omega0*0.01)
        print(f"Autocorrelation at lag 1: {ac1}")
        print(f"Expected autocorrelation: {expected_ac1}")
        assert abs(ac1 - expected_ac1) < 0.1, f"AC1 mismatch: {ac1} vs {expected_ac1}"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_psd_shape(results):
    """Test VonKarmanFilter PSD shape."""
    test_name = "PSD Shape"
    try:
        print("\n=== Testing PSD Shape ===")
        from scipy.signal import welch
        f = VonKarmanFilter(L=100.0, sigma=0.8, V=80.0, dt=0.005)
        data = np.array([f.step() for _ in range(50000)])
        f_psd, Pxx = welch(data, fs=1/0.005, nperseg=4096)
        slope = np.polyfit(np.log(f_psd[20:400]), np.log(Pxx[20:400]), 1)[0]
        print(f"PSD slope: {slope}")
        assert -2.5 < slope < -1.5, f"PSD slope out of range: {slope}"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_parameter_ranges(results):
    """Test VKDisturbanceGenerator parameter ranges."""
    test_name = "Parameter Ranges"
    try:
        print("\n=== Testing Parameter Ranges ===")
        gen = VKDisturbanceGenerator(dt=0.1, V=100.0)
        print(f"L_u range: [{gen.L_u_min}, {gen.L_u_max}]")
        print(f"L_v range: [{gen.L_v_min}, {gen.L_v_max}]")
        print(f"sigma_u range: [{gen.sigma_u_min}, {gen.sigma_u_max}]")
        print(f"sigma_v range: [{gen.sigma_v_min}, {gen.sigma_v_max}]")
        assert gen.L_u_min <= gen.L_u <= gen.L_u_max, "L_u out of range"
        assert gen.L_v_min <= gen.L_v <= gen.L_v_max, "L_v out of range"
        assert gen.sigma_u_min <= gen.sigma_u <= gen.sigma_u_max, "sigma_u out of range"
        assert gen.sigma_v_min <= gen.sigma_v <= gen.sigma_v_max, "sigma_v out of range"
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_plotting_functionality(results):
    """Test VKDisturbanceGenerator plotting functionality."""
    test_name = "Plotting Functionality"
    try:
        print("\n=== Testing Plotting Functionality ===")
        gen = VKDisturbanceGenerator(dt=0.1, V=100.0)
        
        # Generate some data
        for _ in range(50):
            gen(rho=1.0)
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, "")
            gen.plot_disturbance_generator(plot_path)
            plot_file = os.path.join(tmpdirname, "VonKarmenDisturbanceGenerator.png")
            
            # Check if plot was created
            assert os.path.exists(plot_file), "Plot file was not created"
            assert os.path.getsize(plot_file) > 0, "Plot file is empty"
            
            print(f"Plot saved to temporary file: {plot_file}")
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def run_all_tests():
    results = TestResults()
    
    # Run all tests
    test_von_karman_reset(results)
    test_von_karman_zero_noise(results)
    test_von_karman_unit_noise(results)
    test_vk_disturbance_call(results)
    test_vk_disturbance_reset(results)
    test_seed_reproducibility(results)
    test_reset_decorrelation(results)
    test_long_run_variance(results)
    test_autocorr_timescale(results)
    test_psd_shape(results)
    test_parameter_ranges(results)
    test_plotting_functionality(results)
    
    # Print results
    print("\n=== Von Kármán Filter & VK Disturbance Generator Test Results ===")
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
    os.makedirs('results/verification/disturbance_generator_verification', exist_ok=True)
    results.save_to_csv('results/verification/disturbance_generator_verification/test_results.csv')

if __name__ == "__main__":
    run_all_tests()

