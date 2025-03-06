import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def moving_average(var, window_size=5):
    # Calculate moving average
    window_size = min(5, len(var))
    moving_avg = []
    for i in range(len(var)):
        if i < window_size:
            moving_avg.append(sum(var[:i+1]) / (i+1))
        else:
            moving_avg.append(sum(var[i-window_size+1:i+1]) / window_size)
    return moving_avg


def agent_plotter_sac(agent):
    save_path = agent.save_path

    # Plot critic losses
    plt.figure(figsize=(10, 5))
    plt.plot(agent.critic_losses, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    #plt.ylim([0, 50])
    plt.title("Critic Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "critic_losses.png")
    plt.close()

    # Plot actor losses
    plt.figure(figsize=(10, 5))
    plt.plot(agent.actor_losses, label="Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    #plt.ylim([-20, 20])
    plt.title("Actor Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "actor_losses.png")
    plt.close()

    # Plot temperature losses
    plt.figure(figsize=(10, 5))
    plt.plot(agent.temperature_losses, label="Temperature Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Temperature Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "temperature_losses.png")
    plt.close()

    # Plot temperature values
    plt.figure(figsize=(10, 5))
    plt.plot(agent.temperature_values, label="Temperature Value")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Temperature Value")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "temperature_values.png")
    plt.close()

    # Plot number of steps
    plt.figure(figsize=(10, 5))
    plt.plot(agent.number_of_steps, label="Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Number of Steps")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "number_of_steps.png")
    plt.close()


def agent_plotter_sac_marl_ctde(agent):
    save_path = agent.save_path

    # First plot the central agent only
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)

    # Plot central agent metrics (already episode-averaged)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(agent.central_critic_losses, label="Critic Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.set_title("Critic Loss")
    ax1.legend()
    ax1.grid()

    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(agent.central_actor_losses, label="Actor Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Actor Loss")
    ax2.legend()
    ax2.grid()

    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(agent.central_temperature_losses, label="Temperature Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Temperature Loss")
    ax3.legend()
    ax3.grid()

    ax4 = plt.subplot(gs[3, 0])
    ax4.plot(agent.central_number_of_steps, label="Steps")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps")
    ax4.set_title("Number of Steps")
    ax4.legend()
    ax4.grid()

    ax5 = plt.subplot(gs[4, 0])
    ax5.plot(agent.central_temperature_values, label="Temperature Value")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Value")
    ax5.set_title("Temperature Value")
    ax5.legend()
    ax5.grid()

    plt.savefig(save_path + "central_agent_only.png")
    plt.close()

    # Now worker agents only
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

    # Plot worker metrics (already episode-averaged)
    ax1 = plt.subplot(gs[0, 0])
    worker_actor_losses = np.array(agent.actor_loss_workers)
    for i in range(agent.number_of_workers):
        ax1.plot(worker_actor_losses[:, i], label=f"Worker {i+1} Actor Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.set_title("Worker Actor Losses")
    ax1.legend()
    ax1.grid()

    ax2 = plt.subplot(gs[1, 0])
    worker_temp_losses = np.array(agent.temperature_loss_workers)
    for i in range(agent.number_of_workers):
        ax2.plot(worker_temp_losses[:, i], label=f"Worker {i+1} Temperature Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Worker Temperature Losses")
    ax2.legend()
    ax2.grid()

    ax3 = plt.subplot(gs[2, 0])
    worker_temps = np.array(agent.worker_temperature_values)
    for i in range(agent.number_of_workers):
        ax3.plot(worker_temps[:, i], label=f"Worker {i+1} Temperature Value")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Value")
    ax3.set_title("Worker Temperature Values")
    ax3.legend()
    ax3.grid()

    plt.savefig(save_path + "worker_agents_only.png")
    plt.close()

    # Combined plot for steps
    plt.figure(figsize=(20, 15))
    plt.subplot(1, 1, 1)
    for i in range(agent.number_of_workers):
        plt.plot(agent.number_of_steps[i], label=f"Worker {i+1} Steps")
    plt.plot(agent.central_number_of_steps, label="Central Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Number of Steps per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(save_path + "steps_comparison.png")
    plt.close()