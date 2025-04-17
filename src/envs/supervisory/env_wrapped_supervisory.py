import numpy as np

from src.envs.base_environment import rocket_environment_pre_wrap

class supervisory_wrapper:
    def __init__(self,
                 input_normalisation_values,
                 flight_phase = 'subsonic'):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']
        self.flight_phase = flight_phase
        self.input_normalisation_values = input_normalisation_values
        self.env = rocket_environment_pre_wrap(type = 'supervisory',
                                               flight_phase = self.flight_phase)
        self.initial_mass = self.env.reset()[-2]

    def truncation_id(self):
        return self.env.truncation_id

    def augment_state(self, state):# state used will be: [x, y, vx, vy, theta, theta_dot, alpha, mass]
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if self.flight_phase in ['subsonic', 'supersonic']:
            de_normalised_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
        elif self.flight_phase == 'flip_over_boostbackburn':
            de_normalised_state = np.array([theta, theta_dot])
        return de_normalised_state / self.input_normalisation_values
    
    def step(self, action):
        action_numpy = np.array(action)  # Convert JAX array to NumPy array
        state, reward, done, truncated, info = self.env.step(action_numpy)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state