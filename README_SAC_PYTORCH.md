# PyTorch SAC Implementation for Lunar Lander

This repository contains a PyTorch implementation of the Soft Actor-Critic (SAC) algorithm for solving the Lunar Lander continuous control task.

## Features

- Full PyTorch implementation of SAC
- Simple replay buffer for experience storage
- Automatic entropy tuning
- Training and evaluation scripts
- Visualization of trained agents

## Requirements

Make sure you have the following packages installed:

```bash
pip install gymnasium torch numpy matplotlib tqdm
```

## Files Overview

- `src/agents/sac_pytorch.py`: Core implementation of the SAC algorithm using PyTorch
- `validate_sac_pytorch_lunarlander.py`: Script to train and validate the SAC agent on Lunar Lander
- `render_sac_pytorch_lunarlander.py`: Script to visualize a trained agent's performance

## Training the Agent

To train a new agent, run:

```bash
python validate_sac_pytorch_lunarlander.py
```

This will train the agent for 500 episodes, evaluating it every 10 episodes. 
The script will create the following:

- Training progress plots in `results/PyTorchSAC/LunarLander/`
- Saved models in `data/agent_saves/PyTorchSAC/LunarLander/saves/`

## Visualizing a Trained Agent

To visualize a trained agent, run:

```bash
python render_sac_pytorch_lunarlander.py
```

By default, it will look for the most recently saved model in the default save location.
You can also specify a specific model path:

```bash
python render_sac_pytorch_lunarlander.py --model_path data/agent_saves/PyTorchSAC/LunarLander/saves/your_model_name
```

Additional options:
- `--episodes N`: Render N episodes (default: 5)
- `--seed S`: Set a specific random seed

## SAC Algorithm Details

The SAC algorithm is based on the paper:
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) by Haarnoja et al.

Key features of the implementation:

1. **Actor Network**: Outputs mean and log_std of a Gaussian policy
2. **Dual Critics**: Uses two Q-networks to mitigate overestimation bias
3. **Entropy Regularization**: Automatically tunes the temperature parameter
4. **Soft Target Updates**: Uses polyak averaging for stable learning

## Hyperparameter Configuration

The implementation uses the following default hyperparameters for Lunar Lander:

- Hidden layers: 2 layers of 256 neurons for both actor and critic
- Learning rates: 3e-4 for actor, critic, and alpha
- Discount factor (gamma): 0.99
- Soft update coefficient (tau): 0.005
- Initial temperature (alpha): 0.1
- Batch size: 256
- Buffer size: 100,000 transitions

These hyperparameters can be modified in the training script to optimize performance. 