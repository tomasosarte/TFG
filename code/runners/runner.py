import torch as th
import numpy as np

from utils.transition_batch import TransitionBatch
from environments.environment_tsp import EnvironemntTSP

class Runner:
    """
    Implements a single thread runner class.
    """

    def __init__(self, controller, env: EnvironemntTSP, params={}) -> None:
        
        self.controller = controller
        self.env = env
        self.episode_length = params.get('max_episode_length', 1000)
        self.gamma = params.get('gamma', 0.99)

        # Set up current state and time step
        self.state = None
        self.sum_rewards = 0
        self.time_step = 0
    
    def _next_step(self, done : bool = True, next_state = None) -> None:
        """
        Switch to the next time-step and update internal bookeeping.
        """
        if done:
            self.time_step = 0
            self.sum_rewards = 0
            self.state = self.env.reset()
        else: 
            self.time_step += 1
            self.state = next_state
    
    def transition_format(self) -> dict:
        """
        Returns the format of the transitions: A dictionary of (shape, dtype) entries for each key

        Args:
            None

        Returns:
            (dict) Format transitions
        """
        return {
            'action': ((1,), th.float32),
            'state': ((1,), dict),
            'next_state': ((1,), dict),
            'reward': ((1,), th.float32),
            'done': ((1,), th.bool),
            'return': ((1,), th.float32)
        }
    
    def _wrap_transition(self, action, state, next_state, reward, done):
        """
        Wraps the transition in a dictionary.

        Args:
            action (th.Tensor): The action taken.
            state (th.Tensor): The state before the action.
            next_state (th.Tensor): The state after the action.
            reward (th.Tensor): The reward for the action.
            done (th.Tensor): A boolean indicating if the episode is done.

        Returns:
            dict: A dictionary containing the transition.
        """
        return {
            'action': action,
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'done': done,
            'return': th.zeros(1, dtype=th.float32)
        }
    
    def run(self, n_steps : int, transition_buffer: TransitionBatch = None, trim = True, return_dict = None) -> dict:
        """
        Runs n_steps in the environment and stores them in a transition buffer (newly created if none).
        If n_stpes <= 0, stops at the end of the episode.

        Args:
            n_steps (int): The number of steps to run.
            transition_buffer (TransitionBatch): The buffer to store the transitions.
            trim (bool): A boolean indicating if the buffer should be trimmed.
            return_dict (dict): A dictionary to store the return values.
        
        Returns:
            dict: A dictionary containing the buffer, the mean episode reward, the mean episode length, and the number of environment steps.
        """
        my_transition_buffer = TransitionBatch(n_steps if n_steps > 0 else self.episode_length, self.transition_format())
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        max_steps = n_steps if n_steps > 0 else self.episode_length
        for t in range(max_steps):
            # One step in the environment
            action = self.controller.choose_action(self.state)
            state, reward, done, next_state = self.env.step(action)
            my_transition_buffer.add(self._wrap_transition(action, self.state, next_state, reward, done))
            # Compute discounted returns if episode is done or max_steps is reached
            if self.env.elapsed_time == self.episode_length: done = True
            if done or t == (max_steps - 1):
                my_transition_buffer['returns'][t] =  my_transition_buffer['rewards'][t]
                for i in range(t - 1, episode_start - 1, -1):
                    my_transition_buffer['returns'][i] = my_transition_buffer['rewards'][i] + self.gamma * my_transition_buffer['returns'][i + 1]
                
                episode_start = t + 1
            # Remember statistics and advance, potnetially initializing a new episode
            if done:
                episode_lengths.append(self.env.elapsed_time)
                episode_rewards.append(self.sum_rewards)
            self._next_step(done=done, next_state=next_state)
            time += 1
            # If n_steps <= 0, return after one episode, trimmed if specified
            if n_steps <= 0 and done: 
                my_transition_buffer.trim()
                break

        # Add the sampled transitions to the given transition buffer
        transition_buffer = my_transition_buffer if transition_buffer is None else transition_buffer.add(my_transition_buffer)
        if trim: transition_buffer.trim()

        # Return statistics (mean reward, mean length, and environment steps)
        if return_dict is None: return_dict = {}
        return_dict.update({'buffer': transition_buffer,
                            'episode_reward': None if len(episode_rewards) == 0 else np.mean(episode_rewards),
                            'episode_length': None if len(episode_lengths) == 0 else np.mean(episode_lengths),
                            'env_steps': time})
        return return_dict
    
    def run_episode(self, transition_buffer: TransitionBatch = None, trim = True, return_dict = None) -> dict:
        """
        Returns one episode in the enviornment.
        Returns a dictionary containing the transition_buffer, and the episode statistics.

        Args:
            transition_buffer (TransitionBatch): The buffer to store the transitions.
            trim (bool): A boolean indicating if the buffer should be trimmed.
            return_dict (dict): A dictionary to store the return values.
        
        Returns:
            dict: A dictionary containing the buffer, the episode reward, the episode length, and the number of environment steps.
        """

        return self.run(0, transition_buffer, trim, return_dict)


def test_runner():
    env = EnvironemntTSP(10)
    controller = None
    runner = Runner(controller, env)
    runner.run(10)
    runner.run_episode()
    print("Runner test passed.")