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

from src.agents.soft_actor_critic import SoftActorCritic
from src.agents.functions.buffers import PERBuffer

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

def test_soft_actor_critic_initialization(save_folder):
    """Test initialization of SoftActorCritic class"""
    print("\n========== Soft Actor Critic: Initialization Test ==========")
    
    # Test parameters
    state_dim = 3
    action_dim = 2
    flight_phase = "test_phase"
    
    # Initialize SAC
    sac = SoftActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        flight_phase=flight_phase,
        hidden_dim_actor=256,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=256,
        number_of_hidden_layers_critic=2,
        temperature_initial=0.2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=10000,
        trajectory_length=1000,
        batch_size=32,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        temperature_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        temperature_grad_max_norm=1.0,
        max_std=1.0
    )
    
    # Verify initialization
    test_results.add_result(
        "SAC initialization",
        True,
        "SoftActorCritic initialized successfully"
    )
    
    # Verify network dimensions
    actor_params = sac.actor_params
    critic_params = sac.critic_params
    
    # Check actor network structure
    actor_layers = jax.tree_util.tree_leaves(actor_params)
    test_results.add_result(
        "Actor network structure",
        len(actor_layers) > 0,
        f"Actor network has {len(actor_layers)} layers"
    )
    
    # Check critic network structure
    critic_layers = jax.tree_util.tree_leaves(critic_params)
    test_results.add_result(
        "Critic network structure",
        len(critic_layers) > 0,
        f"Critic network has {len(critic_layers)} layers"
    )
    
    # Verify buffer initialization
    test_results.add_result(
        "Buffer initialization",
        isinstance(sac.buffer, PERBuffer),
        "PERBuffer initialized successfully"
    )
    
    # Verify optimizer states
    test_results.add_result(
        "Optimizer states",
        all([
            sac.actor_opt_state is not None,
            sac.critic_opt_state is not None,
            sac.temperature_opt_state is not None
        ]),
        "All optimizer states initialized"
    )
    
    return all([len(actor_layers) > 0, len(critic_layers) > 0, isinstance(sac.buffer, PERBuffer)])

def test_soft_actor_critic_training_cycle(save_folder):
    """Test complete training cycle of SoftActorCritic"""
    print("\n========== Soft Actor Critic: Training Cycle Test ==========")
    
    # Initialize SAC with smaller parameters for testing
    sac = SoftActorCritic(
        state_dim=3,
        action_dim=2,
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
    
    # Generate test data with proper shapes
    rng_key = jax.random.PRNGKey(0)
    states = jax.random.normal(rng_key, (4, 3))  # (batch_size, state_dim)
    actions = jax.random.normal(jax.random.fold_in(rng_key, 1), (4, 2))  # (batch_size, action_dim)
    rewards = jax.random.normal(jax.random.fold_in(rng_key, 2), (4, 1))  # (batch_size, 1)
    next_states = jax.random.normal(jax.random.fold_in(rng_key, 3), (4, 3))  # (batch_size, state_dim)
    dones = jnp.zeros((4, 1))  # (batch_size, 1)
    
    # Store initial parameters
    initial_actor_params = jax.tree_util.tree_map(lambda x: x.copy(), sac.actor_params)
    initial_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), sac.critic_params)
    initial_temperature = sac.temperature
    
    # Ensure proper shapes for concatenation
    states = jnp.reshape(states, (len(states), -1))  # Reshape to (batch_size, state_dim)
    actions = jnp.reshape(actions, (len(actions), -1))  # Reshape to (batch_size, action_dim)
    rewards = jnp.reshape(rewards, (len(rewards), 1))  # Reshape to (batch_size, 1)
    next_states = jnp.reshape(next_states, (len(next_states), -1))  # Reshape to (batch_size, state_dim)
    dones = jnp.reshape(dones, (len(dones), 1))  # Reshape to (batch_size, 1)
    
    # Calculate TD errors in batch using vectorized method
    td_errors = sac.calculate_td_error_vmap(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones
    )
    
    # Add experiences to buffer in batch
    for i in range(len(states)):
        sac.buffer.add(
            state=states[i],
            action=actions[i],
            reward=jnp.squeeze(rewards[i]),
            next_state=next_states[i],
            done=jnp.squeeze(dones[i]),
            td_error=jnp.squeeze(td_errors[i])
        )
    
    # Run critic warm up
    critic_loss = sac.critic_warm_up_step()
    test_results.add_result(
        "Critic warm up step",
        critic_loss is not None,
        f"Critic warm up loss: {critic_loss}"
    )
    
    # Run multiple updates to ensure temperature has a chance to change
    for _ in range(3):  # Run a few updates to ensure temperature changes
        sac.update()
        sac.update_episode()  # Reset first_step_bool flag
    test_results.add_result(
        "Update step",
        True,
        "Update step completed successfully"
    )
    
    # Verify parameter updates
    actor_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            initial_actor_params,
            sac.actor_params
        )
    )
    test_results.add_result(
        "Actor parameters updated",
        actor_params_changed,
        "Actor parameters changed after update"
    )
    
    critic_params_changed = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            initial_critic_params,
            sac.critic_params
        )
    )
    test_results.add_result(
        "Critic parameters updated",
        critic_params_changed,
        "Critic parameters changed after update"
    )
    
    temperature_changed = not jnp.allclose(initial_temperature, sac.temperature, rtol=1e-5)
    test_results.add_result(
        "Temperature updated",
        temperature_changed,
        f"Temperature changed from {initial_temperature} to {sac.temperature}"
    )
    
    # Plot training metrics
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [critic_loss, sac.critic_loss_episode], 'b-o', label='Critic Loss')
    plt.plot([0, 1], [0, sac.actor_loss_episode], 'r-o', label='Actor Loss')
    plt.plot([0, 1], [0, sac.temperature_loss_episode], 'g-o', label='Temperature Loss')
    plt.title('Training Metrics Over Time', fontsize=22)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/training_metrics.png', dpi=300)
    
    return all([critic_loss is not None, actor_params_changed, critic_params_changed, temperature_changed])

def test_soft_actor_critic_action_selection(save_folder):
    """Test action selection methods of SoftActorCritic"""
    print("\n========== Soft Actor Critic: Action Selection Test ==========")
    
    # Initialize SAC
    sac = SoftActorCritic(
        state_dim=3,
        action_dim=2,
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
    
    # Generate test state
    rng_key = jax.random.PRNGKey(0)
    state = jax.random.normal(rng_key, (3,))
    
    # Test stochastic action selection
    actions = sac.select_actions(state)
    test_results.add_result(
        "Stochastic action selection",
        actions.shape == (2,),
        f"Selected actions shape: {actions.shape}"
    )
    
    # Test deterministic action selection
    actions_no_stochastic = sac.select_actions_no_stochastic(state)
    test_results.add_result(
        "Deterministic action selection",
        actions_no_stochastic.shape == (2,),
        f"Selected actions shape: {actions_no_stochastic.shape}"
    )
    
    # Verify actions are different
    actions_different = not jnp.array_equal(actions, actions_no_stochastic)
    test_results.add_result(
        "Stochastic and deterministic actions differ",
        actions_different,
        "Stochastic and deterministic actions are different as expected"
    )
    
    # Plot action distributions
    plt.figure(figsize=(12, 8))
    plt.hist(actions, bins=20, alpha=0.5, label='Stochastic Actions')
    plt.hist(actions_no_stochastic, bins=20, alpha=0.5, label='Deterministic Actions')
    plt.title('Action Distributions', fontsize=22)
    plt.xlabel('Action Value', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/action_distributions.png', dpi=300)
    
    return all([actions.shape == (2,), actions_no_stochastic.shape == (2,), actions_different])

def run_all_tests():
    """Run all SoftActorCritic verification tests"""
    print("\n===== STARTING SOFT ACTOR CRITIC VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/soft_actor_critic_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Initialization
    print("\n[1/3] Running Initialization Test...")
    init_test_result = test_soft_actor_critic_initialization(save_folder)
    
    # Test 2: Training Cycle
    print("\n[2/3] Running Training Cycle Test...")
    training_test_result = test_soft_actor_critic_training_cycle(save_folder)
    
    # Test 3: Action Selection
    print("\n[3/3] Running Action Selection Test...")
    action_test_result = test_soft_actor_critic_action_selection(save_folder)
    
    print("\n===== SOFT ACTOR CRITIC VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/soft_actor_critic_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 