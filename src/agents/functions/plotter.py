import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def agent_plotter_sac(agent):
    save_path = agent.save_path

    # Plot critic losses
    plt.figure(figsize=(20,15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    plt.suptitle('Reinforcement Learning', fontsize = 32)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(agent.critic_losses, label="Critic Loss", linewidth = 4, color = 'blue')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Critic Loss", fontsize = 22)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(agent.actor_losses, label="Actor Loss", linewidth = 4, color = 'blue')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Loss", fontsize = 20)
    ax2.set_title("Actor Loss", fontsize = 22)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(agent.temperature_losses, label="Temperature Loss", linewidth = 4, color = 'blue')
    ax3.set_xlabel("Episode", fontsize = 20)
    ax3.set_ylabel("Loss", fontsize = 20)
    ax3.set_title("Temperature Loss", fontsize = 22)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.grid()

    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(agent.temperature_values, label="Temperature Value", linewidth = 4, color = 'blue')
    ax4.set_xlabel("Episode", fontsize = 20)
    ax4.set_ylabel("Value", fontsize = 20)
    ax4.set_title("Temperature Value", fontsize = 22)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.grid()
    plt.savefig(save_path + "sac_losses.png")
    plt.close()

    # Plot number of steps
    plt.figure(figsize=(10, 5))
    plt.plot(agent.number_of_steps, label="Steps", linewidth = 4, color = 'blue')
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Steps", fontsize=20)
    plt.title("Number of Steps", fontsize=22)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(save_path + "number_of_steps.png")
    plt.close()

def agent_plotter_td3(agent):
    save_path = agent.save_path

    # Plot critic and actor losses
    plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    plt.suptitle('Reinforcement Learning', fontsize = 32)
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(agent.critic_losses, label="Critic Loss", linewidth = 4, color = 'blue')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Critic Loss", fontsize = 22)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(agent.actor_losses, label="Actor Loss", linewidth = 4, color = 'blue')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Loss", fontsize = 20)
    ax2.set_title("Actor Loss", fontsize = 22)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    
    plt.savefig(save_path + "td3_losses.png")
    plt.close()

    # Plot number of steps
    plt.figure(figsize=(10, 5))
    plt.plot(agent.number_of_steps, label="Steps", linewidth = 4, color = 'blue')
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Steps", fontsize=20)
    plt.title("Number of Steps", fontsize=22)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(save_path + "number_of_steps.png")
    plt.close()