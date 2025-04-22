import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

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