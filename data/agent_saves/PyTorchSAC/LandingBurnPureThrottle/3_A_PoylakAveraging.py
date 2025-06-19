import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Function to apply moving average smoothing
def moving_average(data, window_size=20):
    """Apply moving average to data series with the specified window size"""
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

# Function to filter outliers using IQR method
def filter_outliers(data, k=1.5):
    """Filter outliers using the IQR method with factor k"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return np.clip(data, lower_bound, upper_bound)

# Set paths for the four different Polyak averaging coefficient runs
base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "polyak_0.999": "3_A_2",  # Assuming these are the correct directories
    "polyak_0.995": "3_A_1",
    "polyak_0.98": "3_A_3",
    "polyak_0.995_fast": "3_A_0"  # Assuming this is a faster update version
}

# Map for nicer legend labels with LaTeX math notation
legend_labels = {
    "polyak_0.999": r"$\tau$ = 0.001",
    "polyak_0.995": r"$\tau$ = 0.005",
    "polyak_0.98": r"$\tau$ = 0.01",
    "polyak_0.995_fast": r"$\tau$ = 0.1"
}

# Window size for moving average
window_size = 20
# Heavier smoothing for critic losses and training rewards
heavy_window_size = 50

# Episode limit (clip data at 1000 episodes)
episode_limit = 1000

# Set font sizes
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 14

# Create figure with subplots in a 3x2 grid and add extra space at bottom for legend
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle("SAC Polyak Averaging Coefficient Comparison (First 1000 Episodes)", fontsize=TITLE_SIZE)

# Set font sizes for all tick labels
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

colors = ['red', 'blue', 'green', 'purple']  # Four colors for four runs
linestyles = ['-', '--', '-.', ':']

# Plot 1: Critic Loss vs Steps (smoothed) - top left
axes[0, 0].set_title("Critic Loss", fontsize=TITLE_SIZE)
axes[0, 0].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 2: Temperature (α) - top right
axes[0, 1].set_title(r"Temperature ($\alpha$)", fontsize=TITLE_SIZE)
axes[0, 1].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 3: Q-Values with Uncertainty (mean and std) - middle left
axes[1, 0].set_title("Q-Values", fontsize=TITLE_SIZE)
axes[1, 0].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Q-Value", fontsize=LABEL_SIZE)
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 4: Target Q-Values - middle right
axes[1, 1].set_title("Target Q-Values", fontsize=TITLE_SIZE)
axes[1, 1].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Target Q-Value", fontsize=LABEL_SIZE)
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 5: Training Rewards (with uncertainty and log scale) - bottom left
axes[2, 0].set_title("Training Rewards", fontsize=TITLE_SIZE)
axes[2, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[2, 0].set_yscale('log')  # Set log scale for y-axis
axes[2, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 6: Evaluation Rewards (with uncertainty and log scale) - bottom right
axes[2, 1].set_title("Evaluation Rewards", fontsize=TITLE_SIZE)
axes[2, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[2, 1].set_yscale('log')  # Set log scale for y-axis
axes[2, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Store max step values for each run to clip step data
max_steps = {}

# To keep track of legend handles
legend_handles = []
legend_labels_list = []

for i, (label, run_dir) in enumerate(runs.items()):
    # First, determine the max step value from training data for 1000 episodes
    training_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
    if os.path.exists(training_path):
        training_metrics = pd.read_csv(training_path)
        
        # Clip to episode limit
        training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]
        
        if not training_metrics.empty:
            # Calculate total steps up to episode_limit
            max_steps[label] = training_metrics['steps'].sum()
            print(f"{label} - Max steps at episode {episode_limit}: {max_steps[label]}")
    
    # Load learning stats
    learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
    if os.path.exists(learning_stats_path):
        learning_stats = pd.read_csv(learning_stats_path)
        
        # Clip data based on max steps if available
        if label in max_steps:
            learning_stats = learning_stats[learning_stats['step'] <= max_steps[label]]
        
        # Critic loss smoothing
        if 'critic_loss' in learning_stats.columns and not learning_stats['critic_loss'].isnull().all():
            steps = learning_stats['step'].values
            critic_loss = learning_stats['critic_loss'].values
            
            if len(critic_loss) > 0:  # Only proceed if we have valid data points
                # Filter out outliers
                filtered_critic_loss = filter_outliers(critic_loss, k=1.5)
                
                # Apply moving average with heavier smoothing
                smoothed_loss = moving_average(filtered_critic_loss, min(heavy_window_size, len(filtered_critic_loss)))
                
                # Calculate rolling standard deviation
                loss_std = pd.Series(filtered_critic_loss).rolling(window=min(heavy_window_size, len(filtered_critic_loss)), min_periods=1).std().values
                
                # Plot smoothed critic loss
                line, = axes[0, 0].plot(steps, smoothed_loss, 
                           color=colors[i], linestyle=linestyles[i])
                
                # First time we see this label, add to legend handles
                if label not in legend_labels_list:
                    legend_handles.append(line)
                    legend_labels_list.append(legend_labels[label])
                
                # Add uncertainty band (±1 std)
                lower_bound = np.maximum(0, smoothed_loss - loss_std)  # Prevent negative values
                upper_bound = smoothed_loss + loss_std
                
                axes[0, 0].fill_between(
                    steps,
                    lower_bound,
                    upper_bound,
                    color=colors[i], alpha=0.2
                )
        
        # Temperature (α) plotting
        if 'alpha_value' in learning_stats.columns:
            alpha_values = learning_stats['alpha_value'].fillna(0).values
            
            if len(alpha_values) > 0:  # Only proceed if we have valid data points
                # Apply moving average
                smoothed_alpha = moving_average(alpha_values, window_size)
                
                # Plot temperature
                axes[0, 1].plot(
                    steps,
                    smoothed_alpha,
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=2
                )
        
        # Q-Values with uncertainty (mean and std only)
        if 'q_value_mean' in learning_stats.columns and 'q_value_std' in learning_stats.columns:
            # Extract Q-value data
            q_mean = learning_stats['q_value_mean'].fillna(0).values
            q_std = learning_stats['q_value_std'].fillna(0).values
            
            if len(q_mean) > 0:  # Only proceed if we have valid data points
                # Filter out outliers from mean
                q_mean = filter_outliers(q_mean, k=2.0)
                
                # Apply moving average
                smoothed_q_mean = moving_average(q_mean, window_size)
                
                # Plot Q-value mean with uncertainty
                axes[1, 0].fill_between(
                    steps,
                    smoothed_q_mean - q_std,
                    smoothed_q_mean + q_std,
                    color=colors[i],
                    alpha=0.3
                )
                
                # Plot the mean line
                axes[1, 0].plot(
                    steps,
                    smoothed_q_mean,
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=2
                )
        
        # Target Q-Values
        if 'target_q_mean' in learning_stats.columns and 'target_q_std' in learning_stats.columns:
            # Extract Target Q-value data
            target_q_mean = learning_stats['target_q_mean'].fillna(0).values
            target_q_std = learning_stats['target_q_std'].fillna(0).values
            
            if len(target_q_mean) > 0:  # Only proceed if we have valid data points
                # Filter out outliers from mean
                target_q_mean = filter_outliers(target_q_mean, k=2.0)
                
                # Apply moving average
                smoothed_target_q_mean = moving_average(target_q_mean, window_size)
                
                # Plot Target Q-value mean with uncertainty
                axes[1, 1].fill_between(
                    steps,
                    smoothed_target_q_mean - target_q_std,
                    smoothed_target_q_mean + target_q_std,
                    color=colors[i],
                    alpha=0.3
                )
                
                # Plot the mean line
                axes[1, 1].plot(
                    steps,
                    smoothed_target_q_mean,
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=2
                )
    
    # Load training metrics again for plotting
    if os.path.exists(training_path):
        training_metrics = pd.read_csv(training_path)
        
        # Clip to episode limit
        training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]
        
        if not training_metrics.empty:
            # Filter out non-positive values for log scale
            mask = training_metrics['reward'] > 0
            episodes = training_metrics['episode'][mask].values
            rewards = training_metrics['reward'][mask].values
            
            if len(rewards) > 0:  # Only proceed if we have valid data points
                # Apply moving average with heavier smoothing
                smoothed_rewards = moving_average(rewards, min(heavy_window_size, len(rewards)))
                
                # Calculate rolling standard deviation - using log-normal for log scale
                log_rewards = np.log(rewards)
                log_std = pd.Series(log_rewards).rolling(window=min(heavy_window_size, len(rewards)), min_periods=1).std().values
                
                # Plot smoothed training rewards
                axes[2, 0].plot(episodes, smoothed_rewards, 
                            color=colors[i], linestyle=linestyles[i])
                
                # Add uncertainty band (±1 std) - in log scale we need to be careful with bounds
                lower_bound = smoothed_rewards / np.exp(log_std)
                upper_bound = smoothed_rewards * np.exp(log_std)
                
                axes[2, 0].fill_between(
                    episodes,
                    lower_bound,
                    upper_bound,
                    color=colors[i], alpha=0.2
                )
    
    # Load evaluation metrics
    eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
    if os.path.exists(eval_path):
        eval_metrics = pd.read_csv(eval_path)
        
        # Clip to episode limit
        eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]
        
        if not eval_metrics.empty:
            # Filter out non-positive values for log scale
            mask = eval_metrics['eval_reward'] > 0
            eval_episodes = eval_metrics['episode'][mask].values
            eval_rewards = eval_metrics['eval_reward'][mask].values
            
            if len(eval_rewards) > 0:  # Only proceed if we have valid data points
                # Apply moving average
                smoothed_eval_rewards = moving_average(eval_rewards, min(window_size, len(eval_rewards)))
                
                # Calculate rolling standard deviation - using log-normal for log scale
                log_eval_rewards = np.log(eval_rewards)
                log_eval_std = pd.Series(log_eval_rewards).rolling(window=min(window_size, len(eval_rewards)), min_periods=1).std().values
                
                # Plot smoothed evaluation rewards
                axes[2, 1].plot(eval_episodes, smoothed_eval_rewards, 
                            color=colors[i], linestyle=linestyles[i])
                
                # Add uncertainty band (±1 std) - in log scale we need to be careful with bounds
                lower_bound = smoothed_eval_rewards / np.exp(log_eval_std)
                upper_bound = smoothed_eval_rewards * np.exp(log_eval_std)
                
                axes[2, 1].fill_between(
                    eval_episodes,
                    lower_bound,
                    upper_bound,
                    color=colors[i], alpha=0.2
                )

# Add grid to all plots
for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

# Add uncertainty legend to the Q-value plots
q_legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, label='Mean'),
    plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.3, label='Mean±Std')
]
axes[1, 0].legend(handles=q_legend_elements, loc='upper right', fontsize=TICK_SIZE)
axes[1, 1].legend(handles=q_legend_elements, loc='upper right', fontsize=TICK_SIZE)

# Adjust layout before adding legend
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Reduced bottom margin for legend

# Add a single horizontal legend at the bottom using the figure directly
fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=len(legend_handles),  # Adjust column count based on number of runs
    fontsize=TICK_SIZE + 6,  # Increased legend font size
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),  # Position closer to graphs
    columnspacing=6.0,  # More space between columns
    handletextpad=1.0    # More space between lines and text
)

# Save figure
plt.savefig(os.path.join(base_path, "plots/polyak_averaging_comparison.png"), dpi=300)
plt.show()

"""
UNDERSTANDING THE UNCERTAINTY VISUALIZATION:

The shaded regions around each curve represent the statistical uncertainty in the data.
These uncertainty bands show ±1 standard deviation from the smoothed values, calculated
using a rolling window approach (window size = 20 for evaluation rewards, 50 for critic losses
and training rewards).

For log-scale plots (training and evaluation rewards):
- We use a log-normal distribution model to represent uncertainty
- Upper bound = smoothed_value * exp(std_dev)
- Lower bound = smoothed_value / exp(std_dev)
- This creates asymmetric bands appropriate for log-scale data

For linear-scale plots:
- Critic loss: We use a standard normal distribution model with rolling std deviation
  - Upper bound = smoothed_value + std_dev
  - Lower bound = max(0, smoothed_value - std_dev) to prevent negative values
- Q-values and Target Q-values: We use the actual std deviation values from the network's estimates
  - Upper bound = smoothed_mean + q_std
  - Lower bound = smoothed_mean - q_std
  - This represents the network's inherent uncertainty in its value estimates

These uncertainty bands indicate the variability in the measurements:
- Wider bands = higher variability (less consistent performance)
- Narrower bands = lower variability (more consistent performance)
- Approximately 68% of the actual data points should fall within these bands

Comparing the uncertainty bands across different Polyak averaging coefficients helps identify which 
coefficient provides more stable and consistent performance during training.
"""

"""
EXAMPLE SCIENTIFIC ARTICLE FIGURE CAPTION:

Figure X: Polyak averaging coefficient (τ) comparison in Soft Actor-Critic (SAC) reinforcement learning
for the first 1000 episodes of training. The figure presents (a) critic loss, (b) temperature parameter (α),
(c) Q-values, (d) target Q-values, (e) training rewards, and (f) evaluation rewards for four different
Polyak averaging coefficients: τ = 0.001 (red), τ = 0.005 (blue), τ = 0.01 (green), and τ = 0.1 (purple).
Shaded regions represent ±1 standard deviation. For critic loss and rewards, this is calculated
using a moving window of 50 timesteps for critic loss and training rewards, and 20 timesteps for evaluation rewards.
For Q-values and target Q-values, the shaded region represents the network's own uncertainty estimate (std deviation)
in its predictions. Reward values are displayed using a logarithmic scale.
The results demonstrate the impact of target network update speed on learning stability and performance.
"""
