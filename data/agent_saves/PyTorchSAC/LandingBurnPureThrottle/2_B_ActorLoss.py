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

# Set paths for the four different actor learning rate runs
base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "actor_lr_0": "2_B_2",
    "actor_lr_1": "2_B_3",
    "actor_lr_2": "2_B_0",
    "actor_lr_3": "2_B_1" # buen
}

# Map for nicer legend labels with LaTeX math notation
legend_labels = {
    "actor_lr_0": r"Run 0 ($\alpha_\zeta = 4e-3$)",
    "actor_lr_1": r"Run 1 ($\alpha_\zeta = 1e-3$)",
    "actor_lr_2": r"Run 2 ($\alpha_\zeta = 1e-4$)",
    "actor_lr_3": r"Run 3 ($\alpha_\zeta = 1e-5$)"
}

# Window size for moving average
window_size = 20
# Heavier smoothing for training rewards (not for actor loss anymore)
heavy_window_size = 50

# Episode limit (clip data at 1000 episodes)
episode_limit = 1000

# Set font sizes
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 14

# Create figure with subplots in a 2x2 grid and add extra space at bottom for legend
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("SAC Actor Loss and Performance Comparison (First 1000 Episodes)", fontsize=TITLE_SIZE)

# Set font sizes for all tick labels
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

colors = ['red', 'blue', 'green', 'purple']  # Four colors for four runs
linestyles = ['-', '--', '-.', ':']

# Plot 1: Actor Loss vs Steps (raw) - top left
axes[0, 0].set_title("Actor Loss (Raw)", fontsize=TITLE_SIZE)
axes[0, 0].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Actor Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
# max y (-10, 10)
# Plot 2: Alpha-Values with Uncertainty (mean and std) - top right
axes[0, 1].set_title(r"Temperature ($\alpha$)", fontsize=TITLE_SIZE)
axes[0, 1].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 3: Training Rewards (with uncertainty and log scale) - bottom left
axes[1, 0].set_title("Training Rewards", fontsize=TITLE_SIZE)
axes[1, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[1, 0].set_yscale('log')  # Set log scale for y-axis
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 4: Evaluation Rewards (with uncertainty and log scale) - bottom right
axes[1, 1].set_title("Evaluation Rewards", fontsize=TITLE_SIZE)
axes[1, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[1, 1].set_yscale('log')  # Set log scale for y-axis
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

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
        
        # Actor loss no smoothing
        if 'actor_loss' in learning_stats.columns and not learning_stats['actor_loss'].isnull().all():
            steps = learning_stats['step'].values
            actor_loss = learning_stats['actor_loss'].values
            
            if len(actor_loss) > 0:  # Only proceed if we have valid data points
                # Filter out outliers
                filtered_actor_loss = filter_outliers(actor_loss, k=1.5)
                
                # Plot raw actor loss (no smoothing)
                line, = axes[0, 0].plot(steps, filtered_actor_loss, 
                           color=colors[i], linestyle=linestyles[i])
                
                # First time we see this label, add to legend handles
                if label not in legend_labels_list:
                    legend_handles.append(line)
                    legend_labels_list.append(legend_labels[label])
                
                # No uncertainty band for raw data
        
        # Alpha-Values - using alpha_value
        if 'alpha_value' in learning_stats.columns:
            # Extract Alpha-value data
            alpha_mean = learning_stats['alpha_value'].fillna(0).values
            
            if len(alpha_mean) > 0:  # Only proceed if we have valid data points
                # Filter out outliers from mean
                alpha_mean = filter_outliers(alpha_mean, k=2.0)
                
                # Apply moving average
                smoothed_alpha_mean = moving_average(alpha_mean, window_size)
                
                # Plot the mean line only (no uncertainty)
                axes[0, 1].plot(
                    steps,
                    smoothed_alpha_mean,
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
                axes[1, 0].plot(episodes, smoothed_rewards, 
                            color=colors[i], linestyle=linestyles[i])
                
                # Add uncertainty band (±1 std) - in log scale we need to be careful with bounds
                lower_bound = smoothed_rewards / np.exp(log_std)
                upper_bound = smoothed_rewards * np.exp(log_std)
                
                axes[1, 0].fill_between(
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
                axes[1, 1].plot(eval_episodes, smoothed_eval_rewards, 
                            color=colors[i], linestyle=linestyles[i])
                
                # Add uncertainty band (±1 std) - in log scale we need to be careful with bounds
                lower_bound = smoothed_eval_rewards / np.exp(log_eval_std)
                upper_bound = smoothed_eval_rewards * np.exp(log_eval_std)
                
                axes[1, 1].fill_between(
                    eval_episodes,
                    lower_bound,
                    upper_bound,
                    color=colors[i], alpha=0.2
                )

# Add grid to all plots
for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

# Add Alpha-value legend to the Alpha-value plot
alpha_legend_elements = [
    Line2D([0], [0], color='gray', linewidth=2, label='Mean')
]
axes[0, 1].legend(handles=alpha_legend_elements, loc='upper right', fontsize=TICK_SIZE)

# Adjust layout before adding legend
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Reduced bottom margin for legend

# Add a single horizontal legend at the bottom using the figure directly
fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=len(legend_handles),  # Adjust column count based on number of runs
    fontsize=TICK_SIZE + 3,  # Increased legend font size
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),  # Position closer to graphs
    columnspacing=6.0,  # More space between columns
    handletextpad=1.0    # More space between lines and text
)

# Save figure
plt.savefig(os.path.join(base_path, "plots/actor_loss_comparison.png"), dpi=300)
plt.show()

"""
UNDERSTANDING THE UNCERTAINTY VISUALIZATION:

The shaded regions around each curve represent the statistical uncertainty in the data.
These uncertainty bands show ±1 standard deviation from the smoothed values, calculated
using a rolling window approach (window size = 20 for evaluation rewards, 50 for
training rewards).

For actor loss: We plot the raw values after outlier filtering, without smoothing or
uncertainty bands to show the actual variance in the data.

For log-scale plots (training and evaluation rewards):
- We use a log-normal distribution model to represent uncertainty
- Upper bound = smoothed_value * exp(std_dev)
- Lower bound = smoothed_value / exp(std_dev)
- This creates asymmetric bands appropriate for log-scale data

For linear-scale plots:
- Alpha-values: We plot the smoothed mean values without uncertainty bands
  - This represents the direct alpha value measurements across different configurations

These uncertainty bands indicate the variability in the measurements:
- Wider bands = higher variability (less consistent performance)
- Narrower bands = lower variability (more consistent performance)
- Approximately 68% of the actual data points should fall within these bands

Comparing the performance across different actor learning rates helps identify which 
learning rate provides more stable and consistent performance during training.
"""

"""
EXAMPLE SCIENTIFIC ARTICLE FIGURE CAPTION:

Figure X: Actor loss and performance comparison in Soft Actor-Critic (SAC) reinforcement learning
for the first 1000 episodes of training. The figure presents (a) raw actor loss,
(b) Alpha-values, (c) training rewards, and (d) evaluation rewards for four different
actor learning rate configurations. Shaded regions in plots (c) and (d) represent 
±1 standard deviation, calculated using a moving window of 50 timesteps for plot (c), 
and 20 timesteps for plot (d). Plot (b) shows the smoothed mean alpha values for each configuration.
Reward values in plots (c) and (d) are displayed using a logarithmic scale.
The results demonstrate the impact of actor learning rate on learning stability and performance.
"""
