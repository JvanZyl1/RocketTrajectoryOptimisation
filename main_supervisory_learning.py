from src.supervisory_learning.supervisory_learn import SupervisoryLearning
from src.envs.supervisory.agent_load_supervisory import plot_trajectory_supervisory
if __name__ == '__main__':
    plot_trajectory_supervisory(flight_phase='supersonic')
    supervisor = SupervisoryLearning(flight_phase='supersonic')
    supervisor()