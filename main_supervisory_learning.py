from src.supervisory_learning.supervisory_learn import SupervisoryLearning

if __name__ == '__main__':
    supervisor = SupervisoryLearning(flight_phase='subsonic')
    supervisor()