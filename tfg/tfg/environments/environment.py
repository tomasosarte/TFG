import torch as th
from threading import Lock

class Environment:
    """
    Abstract class of an environment.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the environment.

        Args:
            kwargs: Additional arguments.
        
        Returns:
            None
        """
        # Elapsed time in the environment
        self.lock = Lock()
        self.elapsed_time = 0
        self.max_episode_length = 0
        self.state_shape = None

    def copy(self) -> 'Environment':
        """
        Returns a copy of the environment.

        Args:
            None
        
        Returns:
            Environment: A copy of the environment.
        """
        pass
    
    def _reward(self, **kwargs) -> float:
        """
        Returns the reward function from the environment.

        Args:
            **kwargs: Additional arguments.
        
        Returns:
            float: The reward for changing from one state to another.
        """
        pass

    def _get_state(self) -> th.Tensor:
        """
        Returns the current state of the environment.
        
        Args:
            None
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        pass
    
    def reset(self) -> dict:
        """
        Resets the environment to the starting state.

        Args:
            None
        
        Returns:
            None
        """
        self.elapsed_time = 0
        
    def step(self, action: int):
        """
        Takes a step in the environment by visiting the city at the given index.
        
        Args:
            action float: The action done in the enviornment
        
        Returns:
            state th.Tensor: The state of the environment after taking the step.
            reward float: The reward for taking the step.
            done bool: A boolean indicating if the episode is done.
            info dict: A dictionary of additional information.
        """
        pass