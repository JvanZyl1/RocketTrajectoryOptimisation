from src.supervisory_learning.supervisory_learn import SupervisoryLearning
from src.envs.supervisory.agent_load_supervisory import plot_trajectory_supervisory
if __name__ == '__main__':
    supervisory = SupervisoryLearning(flight_phase='ballistic_arc_descent')
    supervisory()
    #plot_trajectory_supervisory(flight_phase='subsonic')