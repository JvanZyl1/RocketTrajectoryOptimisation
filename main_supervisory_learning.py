from src.supervisory_learning.supervisory_learn import SupervisoryLearning
from src.envs.supervisory.agent_load_supervisory import plot_trajectory_supervisory
if __name__ == '__main__':
    #supervisory = SupervisoryLearning(flight_phase='re_entry_burn')
    #supervisory()
    plot_trajectory_supervisory(flight_phase='re_entry_burn')