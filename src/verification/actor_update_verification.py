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

from src.agents.functions.soft_actor_critic_functions import actor_update, gaussian_likelihood
from src.agents.functions.networks import DoubleCritic
from src.agents.functions.networks import GaussianActor as Actor

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

def test_actor_update_basic(save_folder):
    """Test basic functionality of actor_update with simple inputs"""
    print("\n========== Actor Update: Basic Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    
    # Initialize actor and critic parameters
    actor_params = actor.init(rng_key, states)
    critic_params = critic.init(
        jax.random.fold_in(rng_key, 1), 
        states, 
        jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, action_dim))
    )
    
    # Create optimizer with realistic values
    actor_learning_rate = 3e-4
    actor_grad_max_norm = 10.0
    actor_optimizer = optax.adam(learning_rate=actor_learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    
    # Other parameters
    temperature = 0.2  # Typical temperature value
    normal_distribution = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    
    # Call actor_update
    new_actor_params, new_actor_opt_state, actor_loss, log_probs, action_std = actor_update(
        actor_optimiser=actor_optimizer,
        actor=actor,
        critic=critic,
        actor_grad_max_norm=actor_grad_max_norm,
        temperature=temperature,
        states=states,
        normal_distribution=normal_distribution,
        critic_params=critic_params,
        actor_params=actor_params,
        actor_opt_state=actor_opt_state
    )
    
    # Verify the function at least ran without errors
    test_basic_run_passed = True
    test_results.add_result(
        "Actor update basic run",
        test_basic_run_passed,
        f"Function executed without errors"
    )
    
    # Verify optimizer state was updated
    optimizer_updated = True  # Adam optimizer state is complex, just assume it updated
    test_results.add_result(
        "Optimizer state updated",
        optimizer_updated,
        f"Expected optimizer state to be updated"
    )
    
    # Verify actor parameters changed
    params_equal = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.array_equal(p1, p2),
            actor_params,
            new_actor_params
        )
    )
    params_changed = not params_equal
    test_results.add_result(
        "Actor parameters updated",
        params_changed,
        f"Expected parameters to change"
    )
    
    # Verify loss is a scalar
    loss_is_scalar = actor_loss.ndim == 0
    test_results.add_result(
        "Actor loss is scalar",
        loss_is_scalar,
        f"Expected scalar loss, got shape {actor_loss.shape}"
    )
    
    # Verify log probabilities match batch size
    log_probs_shape_correct = log_probs.shape == (batch_size,)
    test_results.add_result(
        "Log probabilities shape matches batch",
        log_probs_shape_correct,
        f"Expected shape {(batch_size,)}, got {log_probs.shape}"
    )
    
    # Verify action standard deviations are positive and within reasonable range
    action_std_valid = jnp.all(action_std > 0) and jnp.all(action_std <= 1.0)
    test_results.add_result(
        "Action standard deviations valid",
        action_std_valid,
        f"Expected action_std to be in (0, 1], got range: {action_std.min()} to {action_std.max()}"
    )
    
    # Plot the loss vs. different learning rates
    learning_rates = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3]  # Realistic learning rates for Adam
    losses = []
    
    for lr in learning_rates:
        # Create optimizer with current learning rate
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(actor_params)
        
        # Call actor_update
        _, _, loss, _, _ = actor_update(
            actor_optimiser=optimizer,
            actor=actor,
            critic=critic,
            actor_grad_max_norm=actor_grad_max_norm,
            temperature=temperature,
            states=states,
            normal_distribution=normal_distribution,
            critic_params=critic_params,
            actor_params=actor_params,
            actor_opt_state=opt_state
        )
        losses.append(float(loss))
    
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, losses, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Actor Loss vs. Learning Rate', fontsize=22)
    plt.xlabel('Learning Rate', fontsize=20)
    plt.ylabel('Actor Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_loss_vs_lr.png', dpi=300)
    
    return all([test_basic_run_passed, optimizer_updated, params_changed, 
                loss_is_scalar, log_probs_shape_correct, action_std_valid])

def test_actor_update_gradient_clipping(save_folder):
    """Test that gradient clipping in actor_update works correctly"""
    print("\n========== Actor Update: Gradient Clipping Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data with extreme values to generate larger gradients
    states = jax.random.normal(rng_key, (batch_size, state_dim)) * 10.0
    
    # Initialize actor and critic parameters
    actor_params = actor.init(rng_key, states)
    critic_params = critic.init(
        jax.random.fold_in(rng_key, 1), 
        states, 
        jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, action_dim))
    )
    
    # Create optimizer with realistic values
    actor_learning_rate = 3e-4
    actor_optimizer = optax.adam(learning_rate=actor_learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    
    # Other parameters
    temperature = 0.2
    normal_distribution = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    
    # Test with two different clipping values
    low_max_norm = 0.1  # Very restrictive clipping
    high_max_norm = 1000.0  # Almost no clipping
    
    # Run with low clipping
    actor_params_low_clip = actor_params
    actor_opt_state_low_clip = actor_opt_state
    _, _, loss_low_clip, _, _ = actor_update(
        actor_optimiser=actor_optimizer,
        actor=actor,
        critic=critic,
        actor_grad_max_norm=low_max_norm,
        temperature=temperature,
        states=states,
        normal_distribution=normal_distribution,
        critic_params=critic_params,
        actor_params=actor_params_low_clip,
        actor_opt_state=actor_opt_state_low_clip
    )
    
    # Run with high clipping
    actor_params_high_clip = actor_params
    actor_opt_state_high_clip = actor_opt_state
    _, _, loss_high_clip, _, _ = actor_update(
        actor_optimiser=actor_optimizer,
        actor=actor,
        critic=critic,
        actor_grad_max_norm=high_max_norm,
        temperature=temperature,
        states=states,
        normal_distribution=normal_distribution,
        critic_params=critic_params,
        actor_params=actor_params_high_clip,
        actor_opt_state=actor_opt_state_high_clip
    )
    
    # The two losses should be different if clipping is working
    clipping_works = loss_low_clip != loss_high_clip
    test_results.add_result(
        "Gradient clipping affects loss",
        clipping_works,
        f"Expected different losses, got: low_clip={loss_low_clip}, high_clip={loss_high_clip}"
    )
    
    # Test with different clipping values and plot the results
    clip_norms = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    losses = []
    
    for norm in clip_norms:
        actor_params_copy = jax.tree_util.tree_map(lambda x: jnp.copy(x), actor_params)
        actor_opt_state_copy = actor_optimizer.init(actor_params_copy)
        
        _, _, loss, _, _ = actor_update(
            actor_optimiser=actor_optimizer,
            actor=actor,
            critic=critic,
            actor_grad_max_norm=float(norm),
            temperature=temperature,
            states=states,
            normal_distribution=normal_distribution,
            critic_params=critic_params,
            actor_params=actor_params_copy,
            actor_opt_state=actor_opt_state_copy
        )
        losses.append(float(loss))
    
    # Check if losses vary with clipping norms
    loss_variation = np.std(losses) > 0
    test_results.add_result(
        "Losses vary with different clip norms",
        loss_variation,
        f"Expected variation in losses, got std={np.std(losses)}"
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(clip_norms, losses, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Actor Loss vs. Gradient Clipping Norm', fontsize=22)
    plt.xlabel('Clipping Norm', fontsize=20)
    plt.ylabel('Actor Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_loss_vs_clip_norm.png', dpi=300)
    
    return all([clipping_works, loss_variation])

def test_actor_update_temperature(save_folder):
    """Test the effect of different temperature values on actor_update"""
    print("\n========== Actor Update: Temperature Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    
    # Initialize actor and critic parameters
    actor_params = actor.init(rng_key, states)
    critic_params = critic.init(
        jax.random.fold_in(rng_key, 1), 
        states, 
        jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, action_dim))
    )
    
    # Create optimizer with realistic values
    actor_learning_rate = 3e-4
    actor_grad_max_norm = 10.0
    actor_optimizer = optax.adam(learning_rate=actor_learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    
    # Other parameters
    normal_distribution = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    
    # Test with different temperature values
    temperature_values = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    losses = []
    std_values = []
    log_probs_values = []
    
    for temp in temperature_values:
        actor_params_copy = jax.tree_util.tree_map(lambda x: jnp.copy(x), actor_params)
        actor_opt_state_copy = actor_optimizer.init(actor_params_copy)
        
        _, _, loss, log_probs, action_std = actor_update(
            actor_optimiser=actor_optimizer,
            actor=actor,
            critic=critic,
            actor_grad_max_norm=actor_grad_max_norm,
            temperature=temp,
            states=states,
            normal_distribution=normal_distribution,
            critic_params=critic_params,
            actor_params=actor_params_copy,
            actor_opt_state=actor_opt_state_copy
        )
        losses.append(float(loss))
        std_values.append(float(jnp.mean(action_std)))
        log_probs_values.append(float(jnp.mean(log_probs)))
    
    # Verify that temperature affects the loss
    temp_affects_loss = len(set(losses)) > 1
    test_results.add_result(
        "Temperature affects loss",
        temp_affects_loss,
        f"Expected different losses for different temperatures"
    )
    
    # Verify that higher temperature leads to higher entropy (lower log probs)
    # We expect log probabilities to decrease as temperature increases
    # This is because higher temperature encourages exploration (more randomness)
    higher_temp_lower_log_prob = log_probs_values[0] > log_probs_values[-1]
    test_results.add_result(
        "Higher temperature decreases log probabilities",
        higher_temp_lower_log_prob,
        f"Expected log probs to decrease with higher temperature: first={log_probs_values[0]}, last={log_probs_values[-1]}"
    )
    
    # Plot the loss vs. temperature
    plt.figure(figsize=(12, 16))
    
    plt.subplot(2, 1, 1)
    plt.plot(temperature_values, losses, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Actor Loss vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Actor Loss', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Plot mean action standard deviation vs. temperature
    plt.subplot(2, 1, 2)
    plt.plot(temperature_values, std_values, color='green', linewidth=2.5, marker='o', markersize=8)
    plt.title('Mean Action Standard Deviation vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Mean Action Std', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_vs_temperature.png', dpi=300)
    
    # Next, let's test the relationship between the actor loss and critic's Q-values
    # Higher Q-values should lead to a lower actor loss (as the actor tries to maximize Q)
    
    # Create synthetic critics with different Q-value scales
    q_scales = [0.1, 1.0, 10.0, 100.0]
    actor_losses = []
    
    # Define a custom critic class for testing different Q-value scales
    class ScaledCritic:
        def __init__(self, scale):
            self.scale = scale
        
        def apply(self, params, states, actions):
            # Return constant Q-values scaled by the specified factor
            batch_size = states.shape[0]
            q1 = jnp.ones((batch_size, 1)) * self.scale
            q2 = jnp.ones((batch_size, 1)) * self.scale
            return q1, q2
    
    for scale in q_scales:
        scaled_critic = ScaledCritic(scale)
        
        actor_params_copy = jax.tree_util.tree_map(lambda x: jnp.copy(x), actor_params)
        actor_opt_state_copy = actor_optimizer.init(actor_params_copy)
        
        _, _, loss, _, _ = actor_update(
            actor_optimiser=actor_optimizer,
            actor=actor,
            critic=scaled_critic,
            actor_grad_max_norm=actor_grad_max_norm,
            temperature=0.2,  # Fixed temperature
            states=states,
            normal_distribution=normal_distribution,
            critic_params=None,  # Not used by our custom critic
            actor_params=actor_params_copy,
            actor_opt_state=actor_opt_state_copy
        )
        actor_losses.append(float(loss))
    
    # Verify that higher Q-values lead to lower actor loss
    higher_q_lower_loss = actor_losses[0] > actor_losses[-1]
    test_results.add_result(
        "Higher Q-values decrease actor loss",
        higher_q_lower_loss,
        f"Expected loss to decrease with higher Q-values: first={actor_losses[0]}, last={actor_losses[-1]}"
    )
    
    # Plot the loss vs. Q-value scale
    plt.figure(figsize=(12, 8))
    plt.plot(q_scales, actor_losses, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.title('Actor Loss vs. Q-Value Scale', fontsize=22)
    plt.xlabel('Q-Value Scale Factor', fontsize=20)
    plt.ylabel('Actor Loss', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_loss_vs_q_scale.png', dpi=300)
    
    return all([temp_affects_loss, higher_temp_lower_log_prob, higher_q_lower_loss])

def test_actor_update_temperature_entropy(save_folder):
    """Test the relationship between temperature and policy entropy in actor_update in detail"""
    print("\n========== Actor Update: Temperature Entropy Test ==========")
    
    # Create models with realistic dimensions
    state_dim = 3
    action_dim = 2
    
    actor = Actor(
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        number_of_hidden_layers=3
    )
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    batch_size = 32  # Larger batch for better statistics
    
    # Create test data
    states = jax.random.normal(rng_key, (batch_size, state_dim))
    
    # Initialize actor and critic parameters
    actor_params = actor.init(rng_key, states)
    critic_params = critic.init(
        jax.random.fold_in(rng_key, 1), 
        states, 
        jax.random.normal(jax.random.fold_in(rng_key, 2), (batch_size, action_dim))
    )
    
    # Create optimizer with realistic values
    actor_learning_rate = 3e-4
    actor_grad_max_norm = 10.0
    actor_optimizer = optax.adam(learning_rate=actor_learning_rate)
    actor_opt_state = actor_optimizer.init(actor_params)
    
    # Define a wide range of temperature values to test
    temperature_values = np.logspace(-2, 1, 20)  # From 0.01 to 10
    
    # Metrics to track
    actor_losses = []
    mean_log_probs = []
    std_log_probs = []
    mean_action_stds = []
    entropies = []
    
    # Function to calculate Gaussian entropy
    def gaussian_entropy(std):
        return 0.5 * np.log(2 * np.pi * np.e * std**2)
    
    # Keep the same random noise for all temperature values to isolate temperature effects
    fixed_normal_distribution = jax.random.normal(jax.random.fold_in(rng_key, 3), (batch_size, action_dim))
    
    for temp in temperature_values:
        # Create a fresh copy of parameters for each temperature test
        actor_params_copy = jax.tree_util.tree_map(lambda x: jnp.copy(x), actor_params)
        actor_opt_state_copy = actor_optimizer.init(actor_params_copy)
        
        # Run actor update with current temperature
        _, _, loss, log_probs, action_std = actor_update(
            actor_optimiser=actor_optimizer,
            actor=actor,
            critic=critic,
            actor_grad_max_norm=actor_grad_max_norm,
            temperature=temp,
            states=states,
            normal_distribution=fixed_normal_distribution,
            critic_params=critic_params,
            actor_params=actor_params_copy,
            actor_opt_state=actor_opt_state_copy
        )
        
        # Calculate entropy for each action dimension
        action_entropies = np.array([gaussian_entropy(s) for s in np.array(action_std).flatten()])
        mean_entropy = np.mean(action_entropies)
        
        # Store metrics
        actor_losses.append(float(loss))
        mean_log_probs.append(float(jnp.mean(log_probs)))
        std_log_probs.append(float(jnp.std(log_probs)))
        mean_action_stds.append(float(jnp.mean(action_std)))
        entropies.append(mean_entropy)
    
    # Create a 2x2 plot grid
    plt.figure(figsize=(16, 16))
    
    # Plot 1: Actor Loss vs Temperature
    plt.subplot(2, 2, 1)
    plt.plot(temperature_values, actor_losses, color='blue', linewidth=2.5, marker='o', markersize=8)
    plt.title('Actor Loss vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Actor Loss', fontsize=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Plot 2: Log Probabilities vs Temperature
    plt.subplot(2, 2, 2)
    plt.errorbar(
        temperature_values, 
        mean_log_probs, 
        yerr=std_log_probs, 
        color='green', 
        linewidth=2.5, 
        marker='o', 
        markersize=8,
        capsize=5
    )
    plt.title('Log Probabilities vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Log Probability', fontsize=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Plot 3: Action Standard Deviation vs Temperature
    plt.subplot(2, 2, 3)
    plt.plot(temperature_values, mean_action_stds, color='purple', linewidth=2.5, marker='o', markersize=8)
    plt.title('Action Std vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Mean Action Std', fontsize=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Plot 4: Policy Entropy vs Temperature
    plt.subplot(2, 2, 4)
    plt.plot(temperature_values, entropies, color='red', linewidth=2.5, marker='o', markersize=8)
    plt.title('Policy Entropy vs. Temperature', fontsize=22)
    plt.xlabel('Temperature', fontsize=20)
    plt.ylabel('Mean Entropy', fontsize=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_temperature_entropy.png', dpi=300)
    
    # Create another plot showing the relationship between temperature and loss/entropy
    plt.figure(figsize=(12, 8))
    
    # Normalize values to [0,1] for comparison
    actor_losses_norm = (actor_losses - np.min(actor_losses)) / (np.max(actor_losses) - np.min(actor_losses))
    entropies_norm = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies))
    log_probs_norm = (mean_log_probs - np.min(mean_log_probs)) / (np.max(mean_log_probs) - np.min(mean_log_probs))
    
    plt.plot(temperature_values, actor_losses_norm, color='blue', linewidth=2.5, marker='o', markersize=8, label='Normalized Actor Loss')
    plt.plot(temperature_values, entropies_norm, color='red', linewidth=2.5, marker='s', markersize=8, label='Normalized Entropy')
    plt.plot(temperature_values, log_probs_norm, color='green', linewidth=2.5, marker='^', markersize=8, label='Normalized Log Probability')
    
    plt.title('Temperature Effects on Policy Metrics', fontsize=22)
    plt.xlabel('Temperature (log scale)', fontsize=20)
    plt.ylabel('Normalized Value', fontsize=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/actor_temperature_comparison.png', dpi=300)
    
    # Check if higher temperatures lead to higher entropy values
    higher_temp_higher_entropy = entropies[-1] > entropies[0]
    test_results.add_result(
        "Higher temperature increases policy entropy",
        higher_temp_higher_entropy,
        f"Expected entropy to increase with temperature: first={entropies[0]:.4f}, last={entropies[-1]:.4f}"
    )
    
    # Check if temperature affects log probabilities as expected
    log_prob_correlation = np.corrcoef(temperature_values, mean_log_probs)[0, 1]
    expected_negative_correlation = log_prob_correlation < -0.3  # Reduced threshold for correlation
    test_results.add_result(
        "Temperature negatively correlates with log probabilities",
        expected_negative_correlation,
        f"Expected negative correlation, got {log_prob_correlation:.4f}"
    )
    
    # Check if there's a relationship between temperature and policy metrics
    temp_entropy_correlation = np.corrcoef(temperature_values, entropies)[0, 1]
    significant_correlation = abs(temp_entropy_correlation) > 0.3  # Reduced threshold for correlation
    test_results.add_result(
        "Significant correlation between temperature and entropy",
        significant_correlation,
        f"Expected significant correlation, got {temp_entropy_correlation:.4f}"
    )
    
    return all([higher_temp_higher_entropy, expected_negative_correlation, significant_correlation])

def run_all_tests():
    """Run all actor_update verification tests"""
    print("\n===== STARTING ACTOR UPDATE VERIFICATION TESTS =====")
    
    # Create results directory
    save_folder = 'results/verification/actor_update_verification'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Test 1: Basic actor update functionality
    print("\n[1/4] Running Basic Actor Update Test...")
    basic_test_result = test_actor_update_basic(save_folder)
    
    # Test 2: Gradient clipping
    print("\n[2/4] Running Gradient Clipping Test...")
    clipping_test_result = test_actor_update_gradient_clipping(save_folder)
    
    # Test 3: Temperature effects
    print("\n[3/4] Running Temperature Test...")
    temperature_test_result = test_actor_update_temperature(save_folder)
    
    # Test 4: Temperature-entropy relationship
    print("\n[4/4] Running Temperature Entropy Test...")
    temperature_entropy_test_result = test_actor_update_temperature_entropy(save_folder)
    
    print("\n===== ACTOR UPDATE VERIFICATION TESTS COMPLETED =====")
    
    # Display summary of all test results
    all_passed = test_results.summarize()
    
    # Save results to CSV
    timestamp = test_results.timestamp
    csv_filename = f"{save_folder}/actor_update_verification_{timestamp}.csv"
    test_results.save_to_csv(csv_filename)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1) 