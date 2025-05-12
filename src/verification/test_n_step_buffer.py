import jax
import jax.numpy as jnp
import numpy as np
import unittest

import sys
sys.path.append(".")

from src.agents.functions.buffers import PERBuffer, compute_n_step_single

class TestNStepBuffer(unittest.TestCase):
    def setUp(self):
        # Common parameters for buffer initialization
        self.gamma = 0.99
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_decay = 1e-3
        self.buffer_size = 1000
        self.state_dim = 4
        self.action_dim = 2
        self.trajectory_length = 3
        self.batch_size = 32
        
        # Create a buffer instance for testing
        self.buffer = PERBuffer(
            gamma=self.gamma,
            alpha=self.alpha,
            beta=self.beta,
            beta_decay=self.beta_decay,
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            trajectory_length=self.trajectory_length,
            batch_size=self.batch_size
        )
        
        # Create a sample trajectory for testing
        self.trajectory = [
            # (state, action, reward, next_state, done, td_error)
            (jnp.ones(self.state_dim), jnp.ones(self.action_dim), 1.0, jnp.ones(self.state_dim) * 2, False, 0.5),
            (jnp.ones(self.state_dim) * 2, jnp.ones(self.action_dim) * 2, 2.0, jnp.ones(self.state_dim) * 3, False, 0.6),
            (jnp.ones(self.state_dim) * 3, jnp.ones(self.action_dim) * 3, 3.0, jnp.ones(self.state_dim) * 4, False, 0.7),
            (jnp.ones(self.state_dim) * 4, jnp.ones(self.action_dim) * 4, 4.0, jnp.ones(self.state_dim) * 5, False, 0.8),
            (jnp.ones(self.state_dim) * 5, jnp.ones(self.action_dim) * 5, 5.0, jnp.ones(self.state_dim) * 6, True, 0.9)
        ]
        
    def test_buffer_initialization(self):
        """Test that the buffer is initialized correctly"""
        self.assertEqual(self.buffer.gamma, self.gamma)
        self.assertEqual(self.buffer.state_dim, self.state_dim)
        self.assertEqual(self.buffer.action_dim, self.action_dim)
        self.assertEqual(self.buffer.trajectory_length, self.trajectory_length)
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(len(self.buffer.episode_buffer), 0)
        
    def test_add_single_transition(self):
        """Test adding a single transition to the buffer"""
        state, action, reward, next_state, done, td_error = self.trajectory[0]
        self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # Check that the transition was added to the buffer
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(len(self.buffer.episode_buffer), 1)
        
        # Check that the episode buffer contains the transition
        episode_state, episode_action, episode_reward, episode_next_state, episode_done = self.buffer.episode_buffer[0]
        self.assertTrue(jnp.array_equal(episode_state, state))
        self.assertTrue(jnp.array_equal(episode_action, action))
        self.assertEqual(episode_reward, reward)
        self.assertTrue(jnp.array_equal(episode_next_state, next_state))
        self.assertEqual(episode_done, done)
        
    def test_add_complete_episode(self):
        """Test adding a complete episode to the buffer"""
        # Add all transitions from the trajectory
        for state, action, reward, next_state, done, td_error in self.trajectory:
            self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # Check that all transitions were added to the buffer
        self.assertEqual(len(self.buffer), len(self.trajectory))
        
        # Check that the episode buffer is reset after the episode ends
        self.assertEqual(len(self.buffer.episode_buffer), 0)
        
    def test_n_step_returns_computation(self):
        """Test that n-step returns are computed correctly"""
        # Add transitions up to but not including the terminal state
        for i in range(len(self.trajectory) - 1):
            state, action, reward, next_state, done, td_error = self.trajectory[i]
            self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # At this point, we should have 4 transitions in the buffer and episode buffer
        self.assertEqual(len(self.buffer), 4)
        self.assertEqual(len(self.buffer.episode_buffer), 4)
        
        # Now add the terminal transition
        state, action, reward, next_state, done, td_error = self.trajectory[-1]
        self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # The episode buffer should be reset
        self.assertEqual(len(self.buffer.episode_buffer), 0)
        
        # Check the n-step returns in the buffer
        # We'll sample from the buffer and verify the rewards
        rng_key = jax.random.PRNGKey(0)
        states, actions, rewards, next_states, dones, indices, weights = self.buffer(rng_key)
        
        # Calculate expected n-step returns manually
        # For trajectory_length=3:
        # Transition 0: R0 + gamma*R1 + gamma^2*R2 = 1 + 0.99*2 + 0.99^2*3 = 1 + 1.98 + 2.9403 = 5.9203
        # Transition 1: R1 + gamma*R2 + gamma^2*R3 = 2 + 0.99*3 + 0.99^2*4 = 2 + 2.97 + 3.9204 = 8.8904
        # Transition 2: R2 + gamma*R3 + gamma^2*R4 = 3 + 0.99*4 + 0.99^2*5 = 3 + 3.96 + 4.9005 = 11.8605
        # Transition 3: R3 + gamma*R4 = 4 + 0.99*5 = 4 + 4.95 = 8.95
        # Transition 4: R4 = 5 (terminal state)
        expected_rewards = [5.9203, 8.8904, 11.8605, 8.95, 5.0]
        
        # Get the actual rewards from the buffer
        actual_rewards = []
        for i in range(len(self.trajectory)):
            transition = self.buffer.buffer[i]
            reward_index = self.state_dim + self.action_dim
            actual_rewards.append(float(transition[reward_index]))
        
        # Compare expected and actual rewards
        for i, (expected, actual) in enumerate(zip(expected_rewards, actual_rewards)):
            self.assertAlmostEqual(expected, actual, places=4, 
                                  msg=f"Reward mismatch at transition {i}: expected {expected}, got {actual}")
                                  
    def test_n_step_returns_with_multiple_episodes(self):
        """Test n-step returns with multiple episodes"""
        # Add first episode
        for state, action, reward, next_state, done, td_error in self.trajectory:
            self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # Create a second episode with different rewards
        second_trajectory = [
            (jnp.ones(self.state_dim) * 10, jnp.ones(self.action_dim) * 10, 10.0, jnp.ones(self.state_dim) * 11, False, 1.0),
            (jnp.ones(self.state_dim) * 11, jnp.ones(self.action_dim) * 11, 11.0, jnp.ones(self.state_dim) * 12, False, 1.1),
            (jnp.ones(self.state_dim) * 12, jnp.ones(self.action_dim) * 12, 12.0, jnp.ones(self.state_dim) * 13, True, 1.2)
        ]
        
        # Add second episode
        for state, action, reward, next_state, done, td_error in second_trajectory:
            self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # Check the total number of transitions in the buffer
        self.assertEqual(len(self.buffer), len(self.trajectory) + len(second_trajectory))
        
        # Calculate expected n-step returns for second episode
        # For trajectory_length=3:
        # Transition 0: R0 + gamma*R1 + gamma^2*R2 = 10 + 0.99*11 + 0.99^2*12 = 10 + 10.89 + 11.7612 = 32.6512
        # Transition 1: R1 + gamma*R2 = 11 + 0.99*12 = 11 + 11.88 = 22.88
        # Transition 2: R2 = 12 (terminal state)
        expected_rewards_second_episode = [32.6512, 22.88, 12.0]
        
        # Get the actual rewards from the buffer for the second episode
        actual_rewards_second_episode = []
        for i in range(len(self.trajectory), len(self.trajectory) + len(second_trajectory)):
            transition = self.buffer.buffer[i]
            reward_index = self.state_dim + self.action_dim
            actual_rewards_second_episode.append(float(transition[reward_index]))
        
        # Compare expected and actual rewards for second episode
        for i, (expected, actual) in enumerate(zip(expected_rewards_second_episode, actual_rewards_second_episode)):
            self.assertAlmostEqual(expected, actual, places=4, 
                                  msg=f"Reward mismatch at transition {i} of second episode: expected {expected}, got {actual}")
    
    def test_buffer_overflow(self):
        """Test that the buffer handles overflow correctly"""
        # Fill the buffer beyond capacity
        for i in range(self.buffer_size + 10):
            state = jnp.ones(self.state_dim) * i
            action = jnp.ones(self.action_dim) * i
            reward = float(i)
            next_state = jnp.ones(self.state_dim) * (i + 1)
            done = (i % 10 == 9)  # Every 10th transition is terminal
            td_error = float(i) / 10.0
            
            self.buffer.add(state, action, reward, next_state, done, td_error)
            
            # Check that the episode buffer is reset after each episode
            if done:
                self.assertEqual(len(self.buffer.episode_buffer), 0)
        
        # Check that the buffer size is capped at buffer_size
        self.assertEqual(len(self.buffer), self.buffer_size)
        
    def test_compute_n_step_single_function(self):
        """Test the compute_n_step_single function directly"""
        # Create a buffer with known transitions
        buf = jnp.array([
            # state (4), action (2), reward (1), next_state (4), done (1)
            jnp.concatenate([jnp.ones(4), jnp.ones(2), jnp.array([1.0]), jnp.ones(4) * 2, jnp.array([0.0])]),
            jnp.concatenate([jnp.ones(4) * 2, jnp.ones(2) * 2, jnp.array([2.0]), jnp.ones(4) * 3, jnp.array([0.0])]),
            jnp.concatenate([jnp.ones(4) * 3, jnp.ones(2) * 3, jnp.array([3.0]), jnp.ones(4) * 4, jnp.array([0.0])]),
            jnp.concatenate([jnp.ones(4) * 4, jnp.ones(2) * 4, jnp.array([4.0]), jnp.ones(4) * 5, jnp.array([1.0])])
        ])
        
        # Test with n=3
        n_step_reward, next_state, done_any = compute_n_step_single(
            buf, self.gamma, self.state_dim, self.action_dim, n=3
        )
        
        # Expected n-step return: R0 + gamma*R1 + gamma^2*R2 = 1 + 0.99*2 + 0.99^2*3 = 5.9203
        expected_reward = 1.0 + self.gamma * 2.0 + self.gamma**2 * 3.0
        self.assertAlmostEqual(float(n_step_reward), expected_reward, places=4)
        
        # The compute_n_step_single function uses a backwards return calculation, 
        # and next_state is the next state from the furthest transition considered
        # (after checking for terminal states)
        expected_next_state = jnp.ones(4) * 4  # This is from the 3rd transition
        
        # Print for debugging
        print(f"Next state shape: {next_state.shape}")
        print(f"Expected state shape: {expected_next_state.shape}")
        print(f"Next state: {next_state}")
        print(f"Expected state: {expected_next_state}")
        
        # Check closeness with higher tolerance
        self.assertTrue(jnp.allclose(next_state, expected_next_state, rtol=1e-4, atol=1e-4),
                       f"Next state {next_state} doesn't match expected {expected_next_state}")
        
        # Expected done is False since none of the first 3 transitions are terminal
        self.assertEqual(float(done_any), 0.0)
        
        # Test with terminal state
        n_step_reward, next_state, done_any = compute_n_step_single(
            buf[1:], self.gamma, self.state_dim, self.action_dim, n=3
        )
        
        # Expected n-step return: R0 + gamma*R1 + gamma^2*R2 = 2 + 0.99*3 + 0.99^2*4 = 8.8904
        expected_reward = 2.0 + self.gamma * 3.0 + self.gamma**2 * 4.0
        self.assertAlmostEqual(float(n_step_reward), expected_reward, places=4)
        
        # Expected done is True since one of the transitions is terminal
        self.assertEqual(float(done_any), 1.0)

if __name__ == "__main__":
    unittest.main() 