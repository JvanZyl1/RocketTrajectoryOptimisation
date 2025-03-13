import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from src.envs.env_endo.main_env_endo import rocket_model_endo_ascent

class endo_ascent_wrapped_EA(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = rocket_model_endo_ascent()
        self.initial_mass = self.env.reset()[-2]
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Define observation space with explicit float32 dtype
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.pi/2], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi/2], dtype=np.float32),
            dtype=np.float32
        )

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        
        # Handle tensors by detaching them before converting to numpy
        if isinstance(x, torch.Tensor):
            return torch.tensor([x.detach(),
                                 y.detach(),
                                 theta.detach(),
                                 theta_dot.detach(),
                                 alpha.detach()], dtype=torch.float32)
        else:
            return np.array([x, y, theta, theta_dot, alpha])
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action_detached = action.detach().numpy()
        else:
            action_detached = action
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        state = self.augment_state(state)
        return state, {}  # Gymnasium requires returning a dict as info

# Create log directory
log_dir = "data/agent_saves/StableBaselines3_sac_endo_ascent/logs"
os.makedirs(log_dir, exist_ok=True)
model_dir = "data/agent_saves/StableBaselines3_sac_endo_ascent/models"
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
env = endo_ascent_wrapped_EA()
env = Monitor(env, log_dir)

# Create the model
model = SAC("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            gradient_steps=-1,
            learning_rate=3e-4,
            buffer_size=300000,
            batch_size = 512,
            gamma=0.99,
            policy_kwargs={"net_arch": [256, 256, 256, 256, 256, 256],
                          "clip_mean": 2.0,
                          "activation_fn": torch.nn.Tanh})

# Create a callback for saving checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="sac_endo_ascent_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Train the model
model.learn(total_timesteps=500000,
            log_interval=100,
            callback=checkpoint_callback)

# Save the final model
model.save(f"{model_dir}/sac_endo_ascent_final")

# Plot results
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    # Plot raw data
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, label="Reward")
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title('Raw Rewards over Time')
    plt.grid(True)
    
    # Plot smoothed data
    if len(x) > 100:  # Only smooth if we have enough data
        window = min(len(x) // 10, 100)  # Dynamic window size
        y_smooth = moving_average(y, window)
        x_smooth = x[window-1:]
        
        plt.subplot(122)
        plt.plot(x_smooth, y_smooth, label=f"Smoothed (window={window})")
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title('Smoothed Rewards over Time')
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.show()

# Plot training results
plot_results(log_dir)

print("Training completed and results plotted!")
print("You can view detailed logs with TensorBoard by running:")
print(f"tensorboard --logdir={log_dir}")

# ep_len_mean : Average length of episodes
# ep_rew_mean : Average reward of episodes
# Episodes : Number of episodes
# fps : Frames per second
# time_elapsed : Time taken to train
# total_timesteps : Total time steps
# actor_loss : Actor loss
# ent_coef : Entropy coefficient
# ent_coef_loss : Entropy coefficient loss
# learning_rate : Learning rate
# n_updates : Number of updates