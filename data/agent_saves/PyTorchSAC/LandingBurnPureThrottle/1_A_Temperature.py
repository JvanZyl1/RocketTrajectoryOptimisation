import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Function to apply moving average smoothing
def moving_average(data, window_size=20):
    """Apply moving average to data series with the specified window size"""
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

# Set paths for the three different alpha learning rates
base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "alpha_lr_0.001": "1_A_2",    # Highest learning rate first
    "alpha_lr_0.0003": "1_A_1",
    "alpha_lr_0.0001": "1_A_3"    # Lowest learning rate last
}

# Map for nicer legend labels with LaTeX math notation - with more spacing
legend_labels = {
    "alpha_lr_0.001": r"$\alpha_{\log(\nu)}$ = 0.001",
    "alpha_lr_0.0003": r"$\alpha_{\log(\nu)}$ = 0.0003",
    "alpha_lr_0.0001": r"$\alpha_{\log(\nu)}$ = 0.0001"
}

# Window size for moving average
window_size = 20
# Heavier smoothing for temperature loss and training rewards
heavy_window_size = 50

# Episode limit (clip data at 1000 episodes)
episode_limit = 1000

# Set font sizes
TITLE_SIZE = 20
LABEL_SIZE = 20
TICK_SIZE = 14

# Create figure with subplots in a 2x2 grid and add extra space at bottom for legend
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Even more height for the legend
fig.suptitle("SAC Temperature Comparison for Different Alpha Learning Rates (First 1000 Episodes)", fontsize=TITLE_SIZE)

# Set font sizes for all tick labels
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
# Enable LaTeX rendering for text
plt.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # Use a good math font

colors = ['red', 'blue', 'green']  # Reordering to match learning rate order
linestyles = ['-', '--', '-.']

# Plot 1: Temperature Loss vs Steps (smoothed) - top left
axes[0, 0].set_title("Temperature Loss", fontsize=TITLE_SIZE)
axes[0, 0].set_xlabel("Steps", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Temperature Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

# Plot 2: Temperature Value vs Steps (smoothed) - top right
axes[0, 1].set_title("Temperature", fontsize=TITLE_SIZE)
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
        
        # Temperature loss smoothing
        if 'alpha_loss' in learning_stats.columns and not learning_stats['alpha_loss'].isnull().all():
            steps = learning_stats['step'].values
            # No need to filter out zero/negative values as we're not using log scale anymore
            alpha_loss = learning_stats['alpha_loss'].values
            
            if len(alpha_loss) > 0:  # Only proceed if we have valid data points
                # Apply moving average with heavier smoothing
                smoothed_loss = moving_average(alpha_loss, min(heavy_window_size, len(alpha_loss)))
                
                # Calculate rolling standard deviation
                loss_std = pd.Series(alpha_loss).rolling(window=min(heavy_window_size, len(alpha_loss)), min_periods=1).std().values
                
                # Plot smoothed temperature loss (without label)
                line, = axes[0, 0].plot(steps, smoothed_loss, 
                           color=colors[i], linestyle=linestyles[i])
                
                # First time we see this label, add to legend handles
                if label not in legend_labels_list:
                    legend_handles.append(line)
                    legend_labels_list.append(legend_labels[label])
                
                # Add uncertainty band (±1 std) - using standard error bands for linear scale
                lower_bound = smoothed_loss - loss_std
                upper_bound = smoothed_loss + loss_std
                
                axes[0, 0].fill_between(
                    steps,
                    lower_bound,
                    upper_bound,
                    color=colors[i], alpha=0.2
                )
        
        # Temperature value smoothing
        if 'alpha_value' in learning_stats.columns and not learning_stats['alpha_value'].isnull().all():
            alpha_value = learning_stats['alpha_value'].fillna(0).values
            
            # Apply moving average
            smoothed_value = moving_average(alpha_value, window_size)
            
            # Calculate rolling standard deviation
            value_std = pd.Series(alpha_value).rolling(window=window_size, min_periods=1).std().values
            
            # Plot smoothed temperature value (without label)
            axes[0, 1].plot(steps, smoothed_value, 
                       color=colors[i], linestyle=linestyles[i])
            
            # Add uncertainty band (±1 std)
            axes[0, 1].fill_between(
                steps,
                np.maximum(0, smoothed_value - value_std),  # Prevent negative values
                smoothed_value + value_std,
                color=colors[i], alpha=0.2
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
                
                # Plot smoothed training rewards (without label)
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
                
                # Plot smoothed evaluation rewards (without label)
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

# Adjust layout before adding legend
plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Reduced bottom margin for legend

# Simply add a single horizontal legend at the bottom using the figure directly
fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=3,  # Force 3 columns
    fontsize=TICK_SIZE + 6,  # Increased legend font size
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),  # Position closer to graphs
    columnspacing=6.0,  # Much more space between columns
    handletextpad=1.0    # More space between lines and text
)

plt.savefig(os.path.join(base_path, "plots/temperature_comparison_1000_episodes.png"), dpi=300)
plt.show()

"""
UNDERSTANDING THE UNCERTAINTY VISUALIZATION:

The shaded regions around each curve represent the statistical uncertainty in the data.
These uncertainty bands show ±1 standard deviation from the smoothed values, calculated
using a rolling window approach (window size = 20).

For log-scale plots (alpha loss and rewards):
- We use a log-normal distribution model to represent uncertainty
- Upper bound = smoothed_value * exp(std_dev)
- Lower bound = smoothed_value / exp(std_dev)
- This creates asymmetric bands appropriate for log-scale data

For linear-scale plots (temperature/alpha value):
- We use a standard normal distribution model
- Upper bound = smoothed_value + std_dev
- Lower bound = max(0, smoothed_value - std_dev)
- This creates symmetric bands while preventing negative values

These uncertainty bands indicate the variability in the measurements:
- Wider bands = higher variability (less consistent performance)
- Narrower bands = lower variability (more consistent performance)
- Approximately 68% of the actual data points should fall within these bands

Comparing the uncertainty bands across different learning rates helps identify which 
learning rate provides more stable and consistent performance during training.
"""

"""
EXAMPLE SCIENTIFIC ARTICLE FIGURE CAPTION:

Figure X: Temperature parameter adaptation in Soft Actor-Critic (SAC) reinforcement learning 
for the first 1000 episodes of training. The figure presents (a) temperature loss, 
(b) temperature parameter value, (c) training rewards, and (d) evaluation rewards for 
three different temperature learning rates: α_{log(α)} = 0.001 (red), 0.0003 (blue), 
and 0.0001 (green). Shaded regions represent ±1 standard deviation, calculated using 
a moving window of 50 timesteps for plots (a) and (c), and 20 timesteps for plots (b) and (d). 
Reward values in plots (c) and (d) are displayed using a logarithmic scale.
"""