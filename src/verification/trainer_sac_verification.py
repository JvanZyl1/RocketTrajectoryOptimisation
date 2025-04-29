import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import csv
from pathlib import Path
import optax

# Add src to path
sys.path.append('.')

from src.trainers.trainers import TrainerSAC
from src.agents.soft_actor_critic import SoftActorCritic
from src.envs.rl.env_wrapped_rl import rl_wrapped_env

# For tracking test results
class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []
        self.results = []  # Store all test results for CSV export
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_result(self, test_name, passed, message=None):
        self.total_tests += 1
        result = {
            'timestamp': self.timestamp,
            'test_name': test_name,
            'passed': passed,
            'message': message or "No details provided"
        }
        self.results.append(result)
        
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.failed_tests.append((test_name, message or "No details provided"))
            print(f"❌ {test_name}: FAILED - {message}")
    
    def summarize(self):
        print("\n===== TEST RESULTS SUMMARY =====")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        if self.failed_tests:
            print("\nFailed tests:")
            for test, message in self.failed_tests:
                print(f"- {test}: {message}")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        return self.passed_tests == self.total_tests
    
    def save_to_csv(self, filepath):
        """Save test results to a CSV file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'test_name', 'passed', 'message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
            
            print(f"\nTest results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            return False

# Initialize global test results tracker
test_results = TestResults()

def test_trainer_sac_initialization(save_folder):
    """Test initialization of TrainerSAC class"""
    print("\n========== Trainer SAC: Initialization Test ==========")
    
    # Create environment
    env = rl_wrapped_env(flight_phase='subsonic', enable_wind=True)
    
    # Create SAC agent
    sac = SoftActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        flight_phase="test_phase",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        temperature_initial=0.2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100,
        trajectory_length=10,
        batch_size=4,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        temperature_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        temperature_grad_max_norm=1.0,
        max_std=1.0
    )
    
    # Initialize trainer
    trainer = TrainerSAC(
        env=env,
        agent=sac,
        flight_phase="test_phase",
        num_episodes=10,
        save_interval=5,
        critic_warm_up_steps=10,
        critic_warm_up_early_stopping_loss=0.0,
        load_buffer_from_experiences_bool=False,
        update_agent_every_n_steps=1,
        priority_update_interval=5
    )
    
    # Verify initialization
    test_results.add_result(
        "Trainer initialization",
        True,
        "TrainerSAC initialized successfully"
    )
    
    # Verify environment and agent references
    test_results.add_result(
        "Environment reference",
        trainer.env == env,
        "Environment reference maintained"
    )
    
    test_results.add_result(
        "Agent reference",
        trainer.agent == sac,
        "Agent reference maintained"
    )
    
    # Verify hyperparameters
    test_results.add_result(
        "Hyperparameters",
        all([
            trainer.num_episodes == 10,
            trainer.save_interval == 5,
            trainer.critic_warm_up_steps == 10,
            trainer.update_agent_every_n_steps == 1,
            trainer.priority_update_interval == 5
        ]),
        "Hyperparameters set correctly"
    )
    
    return True

def test_trainer_sac_training_cycle(save_folder):
    """Test the training cycle of TrainerSAC"""
    print("\n========== Trainer SAC: Training Cycle Test ==========")
    
    # Create environment
    env = rl_wrapped_env(flight_phase='subsonic', enable_wind=True)
    
    # Create SAC agent
    sac = SoftActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        flight_phase="test_phase",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        temperature_initial=0.2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100,
        trajectory_length=10,
        batch_size=4,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        temperature_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        temperature_grad_max_norm=1.0,
        max_std=1.0
    )
    
    # Initialize trainer
    trainer = TrainerSAC(
        env=env,
        agent=sac,
        flight_phase="test_phase",
        num_episodes=2,  # Reduced for testing
        save_interval=1,
        critic_warm_up_steps=2,
        critic_warm_up_early_stopping_loss=0.0,
        load_buffer_from_experiences_bool=False,
        update_agent_every_n_steps=1,
        priority_update_interval=1
    )
    
    # Run training
    try:
        trainer.train()
        test_results.add_result(
            "Training execution",
            True,
            "Training completed successfully"
        )
    except Exception as e:
        test_results.add_result(
            "Training execution",
            False,
            f"Training failed with error: {str(e)}"
        )
        return False
    
    # Verify buffer filling
    non_zero_experiences = int(jnp.sum(jnp.any(trainer.agent.buffer.buffer != 0, axis=1)))
    test_results.add_result(
        "Buffer filling",
        non_zero_experiences > 0,
        f"Buffer contains {non_zero_experiences} experiences"
    )
    
    # Verify critic warm-up
    test_results.add_result(
        "Critic warm-up",
        trainer.agent.critic_warm_up_step_idx > 0,
        f"Critic warm-up completed in {trainer.agent.critic_warm_up_step_idx} steps"
    )
    
    # Verify episode tracking
    test_results.add_result(
        "Episode tracking",
        trainer.agent.episode_idx == 2,
        f"Completed {trainer.agent.episode_idx} episodes"
    )
    
    # Verify loss tracking
    test_results.add_result(
        "Loss tracking",
        len(trainer.agent.critic_losses) > 0 and
        len(trainer.agent.actor_losses) > 0 and
        len(trainer.agent.temperature_losses) > 0,
        "Losses tracked throughout training"
    )
    
    return True

def test_trainer_sac_critic_warm_up(save_folder):
    """Test the critic warm-up functionality"""
    print("\n========== Trainer SAC: Critic Warm-up Test ==========")
    
    # Create environment
    env = rl_wrapped_env(flight_phase='subsonic', enable_wind=True)
    
    # Create SAC agent
    sac = SoftActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        flight_phase="test_phase",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        temperature_initial=0.2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100,
        trajectory_length=10,
        batch_size=4,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        temperature_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        temperature_grad_max_norm=1.0,
        max_std=1.0
    )
    
    # Initialize trainer
    trainer = TrainerSAC(
        env=env,
        agent=sac,
        flight_phase="test_phase",
        num_episodes=1,
        save_interval=1,
        critic_warm_up_steps=5,
        critic_warm_up_early_stopping_loss=0.0,
        load_buffer_from_experiences_bool=False,
        update_agent_every_n_steps=1,
        priority_update_interval=1
    )
    
    # Fill the buffer with some experiences first
    trainer.fill_replay_buffer()
    
    # Store initial critic parameters
    initial_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), sac.critic_params)
    
    # Run critic warm-up
    try:
        trainer.critic_warm_up()
        test_results.add_result(
            "Critic warm-up execution",
            True,
            "Critic warm-up completed successfully"
        )
    except Exception as e:
        test_results.add_result(
            "Critic warm-up execution",
            False,
            f"Critic warm-up failed with error: {str(e)}"
        )
        return False
    
    # Verify critic parameters changed
    critic_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            initial_critic_params,
            sac.critic_params
        )
    )
    test_results.add_result(
        "Critic parameter updates",
        critic_params_changed,
        "Critic parameters should change during warm-up"
    )
    
    # Verify TD errors updated
    non_empty_mask = jnp.any(trainer.agent.buffer.buffer != 0, axis=1)
    indices = jnp.where(non_empty_mask)[0]
    td_errors = jnp.abs(trainer.agent.buffer.priorities[indices])
    test_results.add_result(
        "TD error updates",
        jnp.any(td_errors > 1e-6),  # Check if any TD errors are greater than epsilon
        "TD errors should be non-zero after warm-up"
    )
    
    return True

def run_all_tests():
    """Run all TrainerSAC verification tests"""
    print("\n===== STARTING TRAINER SAC VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/trainer_sac_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Initialization
    print("\n[1/3] Running Initialization Test...")
    init_test_result = test_trainer_sac_initialization(save_folder)
    
    # Test 2: Training Cycle
    print("\n[2/3] Running Training Cycle Test...")
    training_test_result = test_trainer_sac_training_cycle(save_folder)
    
    # Test 3: Critic Warm-up
    print("\n[3/3] Running Critic Warm-up Test...")
    warm_up_test_result = test_trainer_sac_critic_warm_up(save_folder)
    
    print("\n===== TRAINER SAC VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/trainer_sac_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 