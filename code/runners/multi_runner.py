from controllers.controller import Controller
from environments.environment import Environment
from runners.runner import Runner
from utils.transition_batch import TransitionBatch
import numpy as np
import threading

class MultiRunner:
    """ Simple class that runs multiple Runner objects in parallel and merges their outputs. """
    def __init__(self, controller: Controller, env: Environment, params: dict = {}) -> None:
        """
        Initializes the multi-runner object.

        Args:
            controller Controller: The controller used in the training process.
            env Environment: The environment used in the training process.
            params dict: A dictionary containing the parameters for the learner.
        
        Returns:
            None
        """
        self.workers = []
        self.runners = []
        n = params.get('parallel_environments', 1)
        for _ in range(n):
            self.runners.append(Runner(controller=controller, env=env.copy(), params=params))
            
    def transition_format(self) -> dict:
        """ 
        Same transition-format as underlying Runners. 

        Args:
            None
        
        Returns:
            dict: The transition format.
        """
        return self.runners[0].transition_format()
    
    def close(self) -> None:
        """ 
        Closes the underlying environment. Should always when ending an experiment. 
        
        Args:
            None
            
        Returns:
            None
        """
        # Join all workers
        for w in self.workers:
            w.join()
        # Exit all environments
        for r in self.runners:
            r.close()
    
    def fork(self, target, common_args=None, specific_args=None) -> None:
        """ 
        Executes the function "target" on all runners. "common_args" is a dictionary of 
        arguments that are passed to all runners, "specific_args" is a list of 
        dictionaries that contain individual parameters for each runner. 
        
        Args:
            target: The function to be executed.
            common_args: A list of arguments that are passed to all runners.
            specific_args: A list of dictionaries that contain individual parameters for each runner.
        
        Returns:
            None
        """ 
        # Fork all runners
        self.workers = []
        for i, r in enumerate(self.runners):
            r_args = [] if specific_args is None else [arg[i] for arg in specific_args]
            self.workers.append(threading.Thread(target=target, args=(r, *common_args, *r_args)))
            self.workers[-1].start()
        # Join all runners
        for w in self.workers:
            w.join()
    
    def run(self, n_steps: int, transition_buffer: TransitionBatch=None, trim: bool=True):
        """ 
        Runs n_steps, split amongst runners, and stores them in the trainsition_buffer (newly created if None).
        If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
        Returns a dictionary containing the transition_buffer and episode statstics. 
        
        Args:
            n_steps (int): The number of steps to run.
            transition_buffer (TransitionBatch): The buffer to store the transitions.
            trim (bool): A boolean indicating if the buffer should be trimmed.
        
        Returns:
            dict: A dictionary containing the transition_buffer and episode statistics.
        """
        n_steps = n_steps // len(self.runners)
        if transition_buffer is None:
            buffer_len = len(self.runners) * (n_steps if n_steps > 0 else self.runners[0].epi_len)
            transition_buffer = TransitionBatch(buffer_len, self.runners[0].transition_format())
        return_dicts = [{} for _ in self.runners]
        self.fork(target=Runner.run, common_args=(n_steps, transition_buffer, False), specific_args=(return_dicts,))
        if trim: transition_buffer.trim()
        rewards = [d['episode_reward'] for d in return_dicts if d['episode_reward'] is not None]
        lengths = [d['episode_length'] for d in return_dicts if d['episode_reward'] is not None]
        return {'buffer': transition_buffer, 
                'episode_reward': np.mean(rewards) if len(rewards) > 0 else None,
                'episode_length': np.mean(lengths) if len(lengths) > 0 else None,
                'env_steps': len(transition_buffer)}

    def run_episode(self, transition_buffer: TransitionBatch=None, trim: bool=True) -> dict:
        """ 
        Runs one episode in the environemnt. 
        Returns a dictionary containing the transition_buffer and episode statstics. 
        
        Args:
            transition_buffer (TransitionBatch): The buffer to store the transitions.
            trim (bool): A boolean indicating if the buffer should be trimmed.
        
        Returns:
            dict: A dictionary containing the transition_buffer and episode statistics.
        """
        return self.run(0, transition_buffer, trim)