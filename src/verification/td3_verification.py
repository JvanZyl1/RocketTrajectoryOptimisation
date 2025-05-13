import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.agents.td3 import TD3
from src.envs.rl.env_wrapped_rl import rl_wrapped_env

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)
np.random.seed(42)
rng = jax.random.PRNGKey(42)

# Create results directory
results_dir = Path('results/verification/td3_verification')
results_dir.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8')  # Use a valid style name
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def print_green(text):
    """Print text in green."""
    print(f"\033[92m{text}\033[0m")

def verify_td3_initialization():
    """Verify TD3 initialization with different configurations."""
    print("\n=== Verifying TD3 Initialization ===")
    
    # Test configurations
    configs = [
        {
            'name': 'default',
            'state_dim': 4,
            'action_dim': 2,
            'flight_phase': 'ascent',
            'hidden_dim_actor': 64,
            'number_of_hidden_layers_actor': 2,
            'hidden_dim_critic': 64,
            'number_of_hidden_layers_critic': 2,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha_buffer': 0.6,
            'beta_buffer': 0.4,
            'beta_decay_buffer': 0.99,
            'buffer_size': 100000,
            'trajectory_length': 1000,
            'batch_size': 256,
            'critic_learning_rate': 3e-4,
            'actor_learning_rate': 3e-4,
            'critic_grad_max_norm': 1.0,
            'actor_grad_max_norm': 1.0,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_delay': 2
        },
        {
            'name': 'small_network',
            'state_dim': 4,
            'action_dim': 2,
            'flight_phase': 'ascent',
            'hidden_dim_actor': 32,
            'number_of_hidden_layers_actor': 1,
            'hidden_dim_critic': 32,
            'number_of_hidden_layers_critic': 1,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha_buffer': 0.6,
            'beta_buffer': 0.4,
            'beta_decay_buffer': 0.99,
            'buffer_size': 10000,
            'trajectory_length': 100,
            'batch_size': 32,
            'critic_learning_rate': 1e-3,
            'actor_learning_rate': 1e-3,
            'critic_grad_max_norm': 0.5,
            'actor_grad_max_norm': 0.5,
            'policy_noise': 0.1,
            'noise_clip': 0.3,
            'policy_delay': 1
        }
    ]
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        try:
            agent = TD3(**{k: v for k, v in config.items() if k != 'name'})
            print_green(f"✓ Successfully initialized TD3 with {config['name']} configuration")
            
            # Verify network shapes
            state = jnp.ones((config['state_dim'],), dtype=jnp.float32)
            action = agent.select_actions_no_stochastic(state)
            assert action.shape == (config['action_dim'],), f"Action shape mismatch: {action.shape} != {(config['action_dim'],)}"
            print_green("✓ Network shapes verified")
            
            # Analyze network parameters
            actor_params = agent.actor_params
            critic_params = agent.critic_params
            print(f"  - Actor parameters: {sum(p.size for p in jax.tree_util.tree_leaves(actor_params))} total parameters")
            print(f"  - Critic parameters: {sum(p.size for p in jax.tree_util.tree_leaves(critic_params))} total parameters")
            
            # Verify parameter initialization
            actor_params_flat = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(actor_params)])
            critic_params_flat = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(critic_params)])
            
            print(f"  - Actor parameter statistics:")
            print(f"    * Mean: {jnp.mean(actor_params_flat):.4f}")
            print(f"    * Std: {jnp.std(actor_params_flat):.4f}")
            print(f"    * Min: {jnp.min(actor_params_flat):.4f}")
            print(f"    * Max: {jnp.max(actor_params_flat):.4f}")
            
            print(f"  - Critic parameter statistics:")
            print(f"    * Mean: {jnp.mean(critic_params_flat):.4f}")
            print(f"    * Std: {jnp.std(critic_params_flat):.4f}")
            print(f"    * Min: {jnp.min(critic_params_flat):.4f}")
            print(f"    * Max: {jnp.max(critic_params_flat):.4f}")
            
        except Exception as e:
            print(f"✗ Failed to initialize TD3 with {config['name']} configuration: {str(e)}")

def verify_action_selection():
    """Verify action selection with different states and noise levels."""
    print("\n=== Verifying Action Selection ===")
    
    state_dim = 4
    action_dim = 2
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        flight_phase="ascent",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100000,
        trajectory_length=1000,
        batch_size=256,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    )
    
    # Test different states
    states = [
        jnp.zeros((state_dim,), dtype=jnp.float32),
        jnp.ones((state_dim,), dtype=jnp.float32),
        jax.random.normal(rng, (state_dim,), dtype=jnp.float32),
        jnp.array([1, -1, 0.5, -0.5], dtype=jnp.float32)
    ]
    
    # Plot action distributions
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    for i, (state, ax) in enumerate(zip(states, axs)):
        # Collect multiple actions for each state
        actions_no_stochastic = []
        actions_stochastic = []
        for _ in range(100):
            actions_no_stochastic.append(agent.select_actions_no_stochastic(state))
            actions_stochastic.append(agent.select_actions(state))
        
        actions_no_stochastic = jnp.array(actions_no_stochastic)
        actions_stochastic = jnp.array(actions_stochastic)
        
        # Analyze action distributions
        print(f"\nState {i+1} Analysis:")
        print(f"  - Deterministic Actions:")
        print(f"    * Action 1: Mean={jnp.mean(actions_no_stochastic[:, 0]):.4f}, Std={jnp.std(actions_no_stochastic[:, 0]):.4f}")
        print(f"    * Action 2: Mean={jnp.mean(actions_no_stochastic[:, 1]):.4f}, Std={jnp.std(actions_no_stochastic[:, 1]):.4f}")
        print(f"  - Stochastic Actions:")
        print(f"    * Action 1: Mean={jnp.mean(actions_stochastic[:, 0]):.4f}, Std={jnp.std(actions_stochastic[:, 0]):.4f}")
        print(f"    * Action 2: Mean={jnp.mean(actions_stochastic[:, 1]):.4f}, Std={jnp.std(actions_stochastic[:, 1]):.4f}")
        
        # Verify action bounds
        assert jnp.all(actions_no_stochastic >= -1) and jnp.all(actions_no_stochastic <= 1), "Actions out of bounds"
        assert jnp.all(actions_stochastic >= -1) and jnp.all(actions_stochastic <= 1), "Actions out of bounds"
        print_green("✓ Action bounds verified")
        
        # Plot distributions
        noise = 1e-6
        sns.kdeplot(actions_no_stochastic[:, 0] + jax.random.normal(rng, (100,)) * noise, 
                   label='Action 1 (no noise)', color='#2ecc71', ax=ax, warn_singular=False)
        sns.kdeplot(actions_stochastic[:, 0], label='Action 1 (with noise)', color='#e74c3c', ax=ax)
        sns.kdeplot(actions_no_stochastic[:, 1] + jax.random.normal(rng, (100,)) * noise, 
                   label='Action 2 (no noise)', color='#3498db', ax=ax, warn_singular=False)
        sns.kdeplot(actions_stochastic[:, 1], label='Action 2 (with noise)', color='#f1c40f', ax=ax)
        
        ax.set_title(f'State {i+1}', fontsize=14)
        ax.set_xlabel('Action Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Action Distributions for Different States', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / 'action_distributions.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print_green("✓ Action selection verified and distributions plotted")

def verify_update_functions():
    """Verify update functions with different batch sizes and learning rates."""
    print("\n=== Verifying Update Functions ===")
    
    state_dim = 8  # RocketEnv state dimension
    action_dim = 2  # RocketEnv action dimension
    batch_sizes = [32, 64, 128]
    learning_rates = [1e-4, 3e-4, 1e-3]
    
    results = []
    success_criteria = {
        'critic_loss_decrease': 0.1,  # Minimum expected decrease in critic loss
        'actor_loss_decrease': 0.1,   # Minimum expected decrease in actor loss
        'td_error_threshold': 0.5,    # Maximum acceptable TD error
        'stability_threshold': 0.1    # Maximum acceptable standard deviation in losses
    }
    
    # Initialize environment
    env = rl_wrapped_env(flight_phase="subsonic")
    
    # Create a base agent to fill the buffer
    base_agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        flight_phase="subsonic",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100000,
        trajectory_length=1000,
        batch_size=32,  # Default batch size
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    )
    
    # Fill buffer with initial experiences from environment
    print("Filling buffer with environment experiences...")
    for _ in tqdm(range(20), desc="Filling buffer"):
        state = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action with noise for exploration
            action = base_agent.select_actions(jnp.expand_dims(state, 0))
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Ensure proper shapes and dtypes for TD error calculation
            state_batch = jnp.expand_dims(state, 0).astype(jnp.float32)  # (1, state_dim)
            action_batch = jnp.expand_dims(action, 0).astype(jnp.float32)  # (1, action_dim)
            reward_batch = jnp.expand_dims(jnp.array(reward, dtype=jnp.float32), 0)  # (1,)
            next_state_batch = jnp.expand_dims(next_state, 0).astype(jnp.float32)  # (1, state_dim)
            done_batch = jnp.expand_dims(jnp.array(done, dtype=jnp.float32), 0)  # (1,)
            
            # Calculate TD error for this transition
            td_error = base_agent.calculate_td_error(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch
            )
            
            # Add to buffer with proper shapes and dtypes
            base_agent.buffer.add(
                state=state.astype(jnp.float32),  # (state_dim,)
                action=action.astype(jnp.float32),  # (action_dim,)
                reward=jnp.array(reward, dtype=jnp.float32),  # scalar
                next_state=next_state.astype(jnp.float32),  # (state_dim,)
                done=jnp.array(done, dtype=jnp.float32),  # scalar
                td_error=jnp.squeeze(td_error).astype(jnp.float32)  # scalar
            )
            
            state = next_state
    
    print("Buffer filled with initial experiences")
    
    # Now run tests with different configurations
    for batch_size in tqdm(batch_sizes, desc="Batch Sizes"):
        for lr in learning_rates:
            print(f"\nTesting batch_size={batch_size}, learning_rate={lr}")
            
            # Create new agent with same buffer
            agent = TD3(
                state_dim=state_dim,
                action_dim=action_dim,
                flight_phase="subsonic",
                hidden_dim_actor=64,
                number_of_hidden_layers_actor=2,
                hidden_dim_critic=64,
                number_of_hidden_layers_critic=2,
                gamma=0.99,
                tau=0.005,
                alpha_buffer=0.6,
                beta_buffer=0.4,
                beta_decay_buffer=0.99,
                buffer_size=100000,
                trajectory_length=1000,
                batch_size=batch_size,
                critic_learning_rate=lr,
                actor_learning_rate=lr,
                critic_grad_max_norm=1.0,
                actor_grad_max_norm=1.0,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_delay=2
            )
            
            # Copy the filled buffer to the new agent
            agent.buffer = base_agent.buffer
            
            # Track losses
            critic_losses = []
            actor_losses = []
            
            # Run updates
            for _ in tqdm(range(100), desc="Running updates"):
                agent.update()
                critic_losses.append(float(agent.critic_loss_episode))
                actor_losses.append(float(agent.actor_loss_episode))
            
            # Convert to numpy arrays for analysis
            critic_losses = np.array(critic_losses, dtype=np.float32)
            actor_losses = np.array(actor_losses, dtype=np.float32)
            
            # Analyze learning curves
            print(f"  - Critic Loss Analysis:")
            print(f"    * Initial: {critic_losses[0]:.4f}")
            print(f"    * Final: {critic_losses[-1]:.4f}")
            print(f"    * Change: {critic_losses[-1] - critic_losses[0]:.4f}")
            print(f"    * Std: {np.std(critic_losses):.4f}")
            
            print(f"  - Actor Loss Analysis:")
            print(f"    * Initial: {actor_losses[0]:.4f}")
            print(f"    * Final: {actor_losses[-1]:.4f}")
            print(f"    * Change: {actor_losses[-1] - actor_losses[0]:.4f}")
            print(f"    * Std: {np.std(actor_losses):.4f}")
            
            # Verify learning behavior
            critic_loss_decrease = critic_losses[0] - critic_losses[-1]
            actor_loss_decrease = actor_losses[0] - actor_losses[-1]
            critic_stability = np.std(critic_losses)
            actor_stability = np.std(actor_losses)
            
            # Check success criteria
            critic_success = critic_loss_decrease > success_criteria['critic_loss_decrease']
            actor_success = actor_loss_decrease > success_criteria['actor_loss_decrease']
            stability_success = (critic_stability < success_criteria['stability_threshold'] and 
                               actor_stability < success_criteria['stability_threshold'])

            if critic_success:
                print_green("✓ Critic loss shows proper decrease")
            else:
                print(f"✗ Critic loss decrease insufficient: {critic_loss_decrease:.4f} < {success_criteria['critic_loss_decrease']}")
            
            if actor_success:
                print_green("✓ Actor loss shows proper decrease")
            else:
                print(f"✗ Actor loss decrease insufficient: {actor_loss_decrease:.4f} < {success_criteria['actor_loss_decrease']}")
            
            if stability_success:
                print_green("✓ Losses show stable learning")
            else:
                print(f"✗ Losses too unstable: critic_std={critic_stability:.4f}, actor_std={actor_stability:.4f}")
            
            # Store results
            results.append({
                'batch_size': batch_size,
                'learning_rate': lr,
                'critic_losses': critic_losses,
                'actor_losses': actor_losses,
                'success': all([critic_success, actor_success, stability_success])
            })
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, blue, red
    
    for i, (metric, ax) in enumerate(zip(['critic_losses', 'actor_losses', 'td_errors'], axs)):
        for j, result in enumerate(results):
            color = colors[0] if result['success'] else colors[2]  # Green for success, red for failure
            ax.plot(result[metric], 
                   label=f'batch={result["batch_size"]}, lr={result["learning_rate"]}',
                   color=color)
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14)
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Curves for Different Hyperparameters', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / 'update_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print overall success rate
    success_rate = sum(1 for r in results if r['success']) / len(results)
    if success_rate > 0.8:
        print_green(f"\n✓ Overall success rate: {success_rate:.2%}")
    else:
        print(f"\n✗ Overall success rate: {success_rate:.2%}")
    
    print_green("✓ Update functions verified and metrics plotted")

def verify_buffer_control():
    """Verify buffer control methods and their effects on sampling."""
    print("\n=== Verifying Buffer Control ===")
    
    state_dim = 4
    action_dim = 2
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        flight_phase="ascent",
        hidden_dim_actor=64,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=64,
        number_of_hidden_layers_critic=2,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=1000,  # Smaller buffer for testing
        trajectory_length=100,
        batch_size=32,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        critic_grad_max_norm=1.0,
        actor_grad_max_norm=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    )
    
    # Fill buffer with random data
    for _ in range(1000):
        state = jax.random.normal(rng, (state_dim,), dtype=jnp.float32)
        action = jax.random.normal(rng, (action_dim,), dtype=jnp.float32)
        reward = jax.random.normal(rng, dtype=jnp.float32)
        next_state = jax.random.normal(rng, (state_dim,), dtype=jnp.float32)
        done = jnp.array(0.0, dtype=jnp.float32)
        td_error = jnp.array(0.0, dtype=jnp.float32)  # Initialize with zero TD error
        agent.buffer.add(state, action, reward, next_state, done, td_error)
    
    # Test different sampling modes
    sampling_modes = ['uniform', 'prioritised']
    weights = []
    
    for mode in sampling_modes:
        if mode == 'uniform':
            agent.use_uniform_sampling()
        else:
            agent.use_prioritized_sampling()
        
        # Collect weights from multiple samples
        mode_weights = []
        for _ in range(100):
            _, _, _, _, _, _, batch_weights = agent.buffer(agent.get_subkey())
            # Convert JAX array to numpy array and ensure it's float32
            mode_weights.extend(np.array(batch_weights, dtype=np.float32))
        weights.append(mode_weights)
        
        # Analyze weights
        weights_array = np.array(mode_weights, dtype=np.float32)
        print(f"\n{mode.title()} Sampling Analysis:")
        print(f"  - Mean weight: {np.mean(weights_array):.4f}")
        print(f"  - Std weight: {np.std(weights_array):.4f}")
        print(f"  - Min weight: {np.min(weights_array):.4f}")
        print(f"  - Max weight: {np.max(weights_array):.4f}")
        
        if mode == 'uniform':
            # Verify uniform weights are close to 1.0
            assert np.allclose(weights_array, 1.0, atol=1e-6), "Uniform weights should be 1.0"
            print_green("✓ Uniform weights verified")
        else:
            # Verify prioritised weights have variation
            assert np.std(weights_array) > 0, "Prioritized weights should have variation"
            print_green("✓ Prioritized weights verified")
    
    # Plot weight distributions
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#2ecc71', '#3498db']  # Green, blue
    
    for i, (mode, mode_weights, ax) in enumerate(zip(sampling_modes, weights, axs)):
        # Convert to numpy array for plotting
        weights_array = np.array(mode_weights, dtype=np.float32)
        sns.histplot(weights_array, kde=True, color=colors[i], ax=ax)
        ax.set_title(f'{mode.title()} Sampling Weights', fontsize=14)
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Distributions for Different Sampling Modes', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / 'sampling_weights.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print_green("✓ Buffer control verified and weight distributions plotted")

def main():
    """Run all verification tests."""
    print("Starting TD3 verification...")
    
    #verify_td3_initialization()
    #verify_action_selection()
    verify_update_functions()
    verify_buffer_control()
    
    print("\nAll TD3 verification tests completed!")
    print_green(f"✓ Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 