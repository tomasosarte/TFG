import torch as th
import numpy as np

from utils.transition_batch import TransitionBatch
from environments.environment import Environment
from controllers.controller import Controller

class Runner:
    """Implements a single thread runner class."""

    def __init__(self, controller: Controller, env: Environment, params: dict = {}) -> None:
        """
        Initializes the runner object.

        Args:
            controller (Controller): The controller used in the training process.
            env (Environment): The environment used in the training process.
            params (dict): A dictionary containing the parameters for the learner.
        
        Returns:
            None
        """
        self.env = env
        self.controller = controller
        self.epi_len = params.get('max_episode_length', self.env._max_episode_steps)
        self.gamma = params.get('gamma', 0.99)
        self.device = params.get('device', 'cpu')

        # Set up current state and time step
        self.state = None
        self.sum_rewards = 0
        self.time = th.tensor(0, dtype=th.float32)
        self.state_shape = self.env.state_shape
        self._next_step() 

        # Set up transition format
        self.max_nodes_per_graph = params.get('max_nodes_per_graph', 10)
        self.node_dimension = params.get('node_dimension', 2)
    
    def _next_step(self, done : bool = True, next_state = None) -> None:
        """
        Switch to the next time-step and update internal bookeeping.

        Args:
            done (bool): A boolean indicating if the episode is done.
            next_state: The next state in the environment.
        """
        self.time = 0 if done else self.time + 1
        if done:
            self.sum_rewards = 0
            self.state = self.env.reset()
            self.epi_len = self.env._max_episode_steps
        else: self.state = next_state
    
    def transition_format(self) -> dict:
        """
        Returns the format of the transitions: A dictionary of (shape, dtype) entries for each key

        Args:
            None

        Returns:
            (dict) Format of the transitions.
        """
        return {'actions': ((1,), th.long),
                'states': (self.state_shape, th.float32),
                'next_states': (self.state_shape, th.float32),
                'rewards': ((1,), th.float32),
                'dones': ((1,), th.bool),
                'returns': ((1,), th.float32)}
    
    def _wrap_transition(self, 
                        action: th.Tensor, 
                        state: th.Tensor, 
                        next_state: th.Tensor, 
                        reward: th.Tensor, 
                        done: th.Tensor) -> dict:
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
            'actions': action,
            'states': state,
            'next_states': next_state,
            'rewards': reward,
            'dones': done,
            'returns': th.zeros(1, 1, dtype=th.float32)
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
        self.state = self.env.reset()
        self.epi_len = self.env._max_episode_steps
        my_transition_buffer = TransitionBatch(n_steps if n_steps > 0 else self.epi_len, self.transition_format())
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        max_steps = n_steps if n_steps > 0 else self.epi_len
        for t in range(max_steps):
            # One step in the envionment
            action = self.controller.choose_action(self.state)
            state, reward, done, next_state = self.env.step(action)
            self.sum_rewards += reward
            my_transition_buffer.add(self._wrap_transition(action, self.state, next_state, reward, done))
            if self.env.elapsed_time == self.epi_len: done = True
            # Compute discounted returns if episode is done or max_steps is reached
            if done or t == (max_steps - 1):
                my_transition_buffer['returns'][t] = my_transition_buffer['rewards'][t]
                for i in range(t - 1, episode_start - 1, -1):
                    my_transition_buffer['returns'][i] = my_transition_buffer['rewards'][i] + self.gamma * my_transition_buffer['returns'][i + 1]
                episode_start = t + 1
            # Remember statistics and advance (potentially initilaizing a new episode)
            if done:
                episode_lengths.append(self.time + 1)
                episode_rewards.append(self.sum_rewards)
            self._next_step(done=done, next_state=next_state)
            time += 1
        
        # Add the sampled transitions to the given transition buffer
        transition_buffer = my_transition_buffer if transition_buffer is None else transition_buffer.add(my_transition_buffer)
        if trim: transition_buffer.trim()

        # Return statistics (mean reward, mean length, and environment steps)
        if return_dict is None: return_dict = {}
        return_dict.update({'buffer': transition_buffer,
                            'episode_reward': None if len(episode_rewards) == 0 else np.mean([r.cpu() for r in episode_rewards]),
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
