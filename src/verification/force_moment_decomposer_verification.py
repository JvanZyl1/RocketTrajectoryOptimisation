import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import sys
sys.path.append('.')
from src.envs.rockets_physics import (
    force_moment_decomposer_ascent,
    force_moment_decomposer_flipoverboostbackburn,
    force_moment_decomposer_re_entry_landing_burn,
    ACS,
    RCS
)

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
            print(f"✅ {test_name} passed")
        else:
            self.failed_tests += 1
            print(f"❌ {test_name} failed: {details}")
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

def test_ascent_decomposer(results):
    """Test the ascent force moment decomposer with various conditions."""
    test_name = "Ascent Decomposer"
    try:
        print("\n=== Testing Ascent Decomposer ===")
        # Test parameters
        atmospheric_pressure = 101325.0  # Pa
        d_thrust_cg = 5.0  # m
        thrust_per_engine = 1000000.0  # N
        nozzle_exit_pressure = 200000.0  # Pa
        nozzle_exit_area = 1.0  # m²
        v_exhaust = 3000.0  # m/s
        
        # Test case 1: Zero gimbal, full throttle
        print("\nTest case 1: Zero gimbal, full throttle")
        actions = jnp.array([0.0, 1.0])
        thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle = force_moment_decomposer_ascent(
            actions=actions,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_gimballed=3,
            number_of_engines_non_gimballed=3,
            v_exhaust=v_exhaust
        )
        
        print(f"Outputs: thrust_parallel={thrust_parallel}, thrust_perpendicular={thrust_perpendicular}, "
              f"moment_z={moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}, throttle={throttle}")
        
        # Verify outputs
        assert jnp.allclose(gimbal_angle_deg, 0.0), "Gimbal angle should be zero"
        assert jnp.allclose(throttle, 1.0), "Throttle should be 1.0"
        assert jnp.allclose(thrust_perpendicular, 0.0), "Perpendicular thrust should be zero"
        assert jnp.allclose(moment_z, 0.0), "Moment should be zero"
        assert thrust_parallel > 0, "Parallel thrust should be positive"
        
        # Test case 2: Maximum gimbal, half throttle
        print("\nTest case 2: Maximum gimbal, half throttle")
        actions = jnp.array([1.0, 0.0])
        thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg, throttle = force_moment_decomposer_ascent(
            actions=actions,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_gimballed=3,
            number_of_engines_non_gimballed=3,
            v_exhaust=v_exhaust
        )
        
        print(f"Outputs: thrust_parallel={thrust_parallel}, thrust_perpendicular={thrust_perpendicular}, "
              f"moment_z={moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}, throttle={throttle}")
        
        # Verify outputs
        assert jnp.allclose(throttle, 0.75), "Throttle should be 0.75 for zero input"
        assert thrust_perpendicular != 0, "Perpendicular thrust should be non-zero"
        assert moment_z != 0, "Moment should be non-zero"
        assert thrust_parallel > 0, "Parallel thrust should be positive"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_flipover_boostback_decomposer(results):
    """Test the flip-over boostback force moment decomposer with various conditions."""
    test_name = "Flip-over Boostback Decomposer"
    try:
        print("\n=== Testing Flip-over Boostback Decomposer ===")
        # Test parameters
        atmospheric_pressure = 101325.0
        d_thrust_cg = 5.0
        thrust_per_engine = 1000000.0
        nozzle_exit_pressure = 200000.0
        nozzle_exit_area = 1.0
        v_exhaust = 3000.0
        dt = 0.01
        
        # Test case 1: Zero command, previous angle zero
        print("\nTest case 1: Zero command, previous angle zero")
        action = 0.0
        gimbal_angle_deg_prev = 0.0
        thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg = force_moment_decomposer_flipoverboostbackburn(
            action=action,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            gimbal_angle_deg_prev=gimbal_angle_deg_prev,
            dt=dt,
            max_gimbal_angle_deg=45.0,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_flip_over_boostbackburn=3,
            v_exhaust=v_exhaust
        )
        
        print(f"Outputs: thrust_parallel={thrust_parallel}, thrust_perpendicular={thrust_perpendicular}, "
              f"moment_z={moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}")
        
        # Verify outputs
        assert jnp.allclose(gimbal_angle_deg, 0.0), "Gimbal angle should be zero"
        assert jnp.allclose(thrust_perpendicular, 0.0), "Perpendicular thrust should be zero"
        assert jnp.allclose(moment_z, 0.0), "Moment should be zero"
        assert thrust_parallel > 0, "Parallel thrust should be positive"
        
        # Test case 2: Maximum command, previous angle zero
        print("\nTest case 2: Maximum command, previous angle zero")
        action = 1.0
        thrust_parallel, thrust_perpendicular, moment_z, mass_flow, gimbal_angle_deg = force_moment_decomposer_flipoverboostbackburn(
            action=action,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            gimbal_angle_deg_prev=gimbal_angle_deg_prev,
            dt=dt,
            max_gimbal_angle_deg=45.0,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_flip_over_boostbackburn=3,
            v_exhaust=v_exhaust
        )
        
        print(f"Outputs: thrust_parallel={thrust_parallel}, thrust_perpendicular={thrust_perpendicular}, "
              f"moment_z={moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}")
        
        # Verify outputs
        assert gimbal_angle_deg > 0, "Gimbal angle should be positive"
        assert thrust_perpendicular != 0, "Perpendicular thrust should be non-zero"
        assert moment_z != 0, "Moment should be non-zero"
        assert thrust_parallel > 0, "Parallel thrust should be positive"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_re_entry_landing_decomposer(results):
    """Test the re-entry landing force moment decomposer with various conditions."""
    test_name = "Re-entry Landing Decomposer"
    try:
        print("\n=== Testing Re-entry Landing Decomposer ===")
        # Test parameters
        atmospheric_pressure = 101325.0
        d_thrust_cg = 5.0
        thrust_per_engine = 1000000.0
        nozzle_exit_pressure = 200000.0
        nozzle_exit_area = 1.0
        v_exhaust = 3000.0
        dt = 0.01
        
        # Test case 1: Zero commands, zero angles
        print("\nTest case 1: Zero commands, zero angles")
        actions = jnp.array([0.0, 0.0])
        pitch_angle = 0.0
        flight_path_angle = 0.0
        dynamic_pressure = 10000.0
        x_cog = 0.0
        delta_command_rad_prev = 0.0
        gimbal_angle_deg_prev = 0.0
        
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad = force_moment_decomposer_re_entry_landing_burn(
            actions=actions,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=delta_command_rad_prev,
            gimbal_angle_deg_prev=gimbal_angle_deg_prev,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_gimballed=3,
            v_exhaust=v_exhaust,
            grid_fin_area=1.0,
            CN_alpha=0.1,
            CN_0=0.01,
            CA_alpha=0.1,
            CA_0=0.01,
            d_base_grid_fin=5.0,
            nominal_throttle=0.5,
            dt=dt
        )
        
        print(f"Outputs: control_force_parallel={control_force_parallel}, control_force_perpendicular={control_force_perpendicular}, "
              f"control_moment_z={control_moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}, "
              f"throttle={throttle}, delta_rad={delta_rad}")
        
        # Verify outputs
        assert jnp.allclose(gimbal_angle_deg, 0.0), "Gimbal angle should be zero"
        assert jnp.allclose(throttle, 0.75), "Throttle should be 0.75 for zero input"
        assert jnp.allclose(delta_rad, 0.0), "Delta rad should be zero"
        assert control_force_parallel > 0, "Control force parallel should be positive"
        
        # Test case 2: Maximum commands
        print("\nTest case 2: Maximum commands")
        actions = jnp.array([1.0, 1.0])
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow, gimbal_angle_deg, throttle, delta_rad = force_moment_decomposer_re_entry_landing_burn(
            actions=actions,
            atmospheric_pressure=atmospheric_pressure,
            d_thrust_cg=d_thrust_cg,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=delta_command_rad_prev,
            gimbal_angle_deg_prev=gimbal_angle_deg_prev,
            thrust_per_engine_no_losses=thrust_per_engine,
            nozzle_exit_pressure=nozzle_exit_pressure,
            nozzle_exit_area=nozzle_exit_area,
            number_of_engines_gimballed=3,
            v_exhaust=v_exhaust,
            grid_fin_area=1.0,
            CN_alpha=0.1,
            CN_0=0.01,
            CA_alpha=0.1,
            CA_0=0.01,
            d_base_grid_fin=5.0,
            nominal_throttle=0.5,
            dt=dt
        )
        
        print(f"Outputs: control_force_parallel={control_force_parallel}, control_force_perpendicular={control_force_perpendicular}, "
              f"control_moment_z={control_moment_z}, mass_flow={mass_flow}, gimbal_angle_deg={gimbal_angle_deg}, "
              f"throttle={throttle}, delta_rad={delta_rad}")
        
        # Verify outputs
        assert gimbal_angle_deg != 0, "Gimbal angle should be non-zero"
        assert throttle > 0.5, "Throttle should be above nominal"
        assert control_force_perpendicular != 0, "Control force perpendicular should be non-zero"
        assert control_moment_z != 0, "Control moment should be non-zero"

        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_acs(results):
    """Test the Attitude Control System with various conditions."""
    test_name = "ACS"
    try:
        print("\n=== Testing ACS ===")
        # Test parameters
        grid_fin_area = 1.0
        CN_alpha = -3.0
        CN_0 = 0.2
        CA_alpha = 0.4
        CA_0 = 0.0
        d_base_grid_fin = 75.40686682968082
        dt = 0.01
        
        # Test case 1: Zero deflection
        print("\nTest case 1: Zero deflection")
        deflection_command_deg = 0.0
        pitch_angle = math.radians(90)
        flight_path_angle = math.radians(170)
        dynamic_pressure = 10000.0
        x_cog =35.0
        delta_command_rad_prev = 0.0
        
        force_parallel, force_perpendicular, moment_z, delta_rad = ACS(
            deflection_command_deg=deflection_command_deg,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=delta_command_rad_prev,
            dt=dt,
            grid_fin_area=grid_fin_area,
            CN_alpha=CN_alpha,
            CN_0=CN_0,
            CA_alpha=CA_alpha,
            CA_0=CA_0,
            d_base_grid_fin=d_base_grid_fin
        )
        
        print(f"Outputs: force_parallel={force_parallel}, force_perpendicular={force_perpendicular}, "
              f"moment_z={moment_z}, delta_rad={delta_rad}")
        
        # Verify outputs
        assert jnp.allclose(delta_rad, 0.0), "Delta rad should be zero"
        assert force_perpendicular != 0, "Force perpendicular should be non-zero due to CN_0 and CA_0"
        assert moment_z != 0, "Moment should be non-zero due to CN_0 and CA_0"
        assert force_parallel != 0, "Force parallel should be non-zero due to CN_0 and CA_0"
        
        # Test case 2: Maximum deflection
        print("\nTest case 2: Maximum deflection")
        deflection_command_deg = 60.0
        force_parallel, force_perpendicular, moment_z, delta_rad = ACS(
            deflection_command_deg=deflection_command_deg,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=delta_command_rad_prev,
            dt=dt,
            grid_fin_area=grid_fin_area,
            CN_alpha=CN_alpha,
            CN_0=CN_0,
            CA_alpha=CA_alpha,
            CA_0=CA_0,
            d_base_grid_fin=d_base_grid_fin
        )
        
        print(f"Outputs: force_parallel={force_parallel}, force_perpendicular={force_perpendicular}, "
              f"moment_z={moment_z}, delta_rad={delta_rad}")
        
        # Verify outputs
        assert delta_rad != 0, "Delta rad should be non-zero"
        assert force_perpendicular != 0, "Force perpendicular should be non-zero"
        assert moment_z != 0, "Moment should be non-zero"
        assert force_parallel > 0, "Force parallel should be positive"

        print(f'\n Test case 3: Zero effective angle of attack')
        pitch_angle = math.radians(110)
        flight_path_angle = pitch_angle + math.pi
        force_parallel, force_perpendicular, moment_z, delta_rad = ACS(
            deflection_command_deg=0.0,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=0.0,
            dt=dt,
            grid_fin_area=grid_fin_area,
            CN_alpha=CN_alpha,
            CN_0=CN_0,
            CA_alpha=CA_alpha,
            CA_0=CA_0,
            d_base_grid_fin=d_base_grid_fin,
            zero_effective_aoa_test_case=True
        )
        
        print(f"Tested within function itself too")
        print(f"Outputs: force_parallel={force_parallel}, force_perpendicular={force_perpendicular}, "
              f"moment_z={moment_z}, delta_rad={delta_rad}")
        
        print(f'\n Test case 4: zero effective angle of attack and minimum deflection i.e. max negative deflection')
        deflection_command_deg = -60.0
        force_parallel, force_perpendicular, moment_z, delta_rad = ACS(
            deflection_command_deg=deflection_command_deg,
            pitch_angle=pitch_angle,
            flight_path_angle=flight_path_angle,
            dynamic_pressure=dynamic_pressure,
            x_cog=x_cog,
            delta_command_rad_prev=0.0,
            dt=dt,
            grid_fin_area=grid_fin_area,
            CN_alpha=CN_alpha,
            CN_0=CN_0,
            CA_alpha=CA_alpha,
            CA_0=CA_0,
            d_base_grid_fin=d_base_grid_fin)
        
        print(f"Outputs: force_parallel={force_parallel}, force_perpendicular={force_perpendicular}, "
              f"moment_z={moment_z}, delta_rad={delta_rad}")
        
        assert force_parallel > 0, "Force parallel should be positive"
        assert force_perpendicular < 0, "Force perpendicular should be negative"
        assert moment_z < 0, "Moment should be negative"
        assert delta_rad < 0, "Delta rad should be negative"
        
        results.add_result(test_name, True)

    except Exception as e:
        results.add_result(test_name, False, str(e))

def test_rcs(results):
    """Test the Reaction Control System with various conditions."""
    test_name = "RCS"
    try:
        print("\n=== Testing RCS ===")
        # Test parameters
        max_RCS_force = 1000.0
        d_base_rcs_bottom = 5.0
        d_base_rcs_top = 5.0
        
        # Test case 1: Zero command
        print("\nTest case 1: Zero command")
        action = jnp.array([0.0])
        x_cog = 0.0
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow = RCS(
            action=action,
            x_cog=x_cog,
            max_RCS_force_per_thruster=max_RCS_force,
            d_base_rcs_bottom=d_base_rcs_bottom,
            d_base_rcs_top=d_base_rcs_top
        )
        
        print(f"Outputs: control_force_parallel={control_force_parallel}, control_force_perpendicular={control_force_perpendicular}, "
              f"control_moment_z={control_moment_z}, mass_flow={mass_flow}")
        
        # Verify outputs
        assert jnp.allclose(control_force_parallel, 0.0), f"Control force parallel should be zero : {control_force_parallel}"
        assert jnp.allclose(control_force_perpendicular, 0.0), f"Control force perpendicular should be zero : {control_force_perpendicular}"
        assert jnp.allclose(control_moment_z, 0.0), f"Control moment should be zero : {control_moment_z}"
        assert jnp.allclose(mass_flow, 0.0), f"Mass flow should be zero : {mass_flow}"
        
        # Test case 2: Maximum command
        print("\nTest case 2: Maximum command")
        action = 1.0
        control_force_parallel, control_force_perpendicular, control_moment_z, mass_flow = RCS(
            action=action,
            x_cog=x_cog,
            max_RCS_force_per_thruster=max_RCS_force,
            d_base_rcs_bottom=d_base_rcs_bottom,
            d_base_rcs_top=d_base_rcs_top
        )
        
        print(f"Outputs: control_force_parallel={control_force_parallel}, control_force_perpendicular={control_force_perpendicular}, "
              f"control_moment_z={control_moment_z}, mass_flow={mass_flow}")
        
        # Verify outputs
        assert control_force_parallel == 0, f"Control force parallel should be equal to 0 : {control_force_parallel}"
        assert control_force_perpendicular == 0, f"Control force perpendicular should be equal to 0 : {control_force_perpendicular}"
        assert control_moment_z > 0, f"Control moment should be equal to 0 : {control_moment_z}"
        assert mass_flow == 0, f"Mass flow should be equal to 0 : {mass_flow}"
        
        results.add_result(test_name, True)
    except Exception as e:
        results.add_result(test_name, False, str(e))

def run_all_tests():
    """Run all verification tests and save results."""
    results = TestResults()
    
    print("\n=== Starting Force Moment Decomposer Verification Tests ===\n")
    
    # Run tests
    test_ascent_decomposer(results)
    test_flipover_boostback_decomposer(results)
    test_re_entry_landing_decomposer(results)
    test_acs(results)
    test_rcs(results)
    
    # Print results
    print(f"\n=== Test Results ===")
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    
    # Save results to CSV
    results_dir = Path("results/verification/force_moment_decomposer_verification")
    results_dir.mkdir(parents=True, exist_ok=True)
    results.save_to_csv(results_dir / "test_results.csv")
    
    return results.failed_tests == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 