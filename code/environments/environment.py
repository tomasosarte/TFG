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

    def _reward(self, current_city: int, previous_city: int) -> float:
        """
        Returns the reward for visiting the current city from the previous city.

        Args:
            current_city (int): The index of the current city.
            previous_city (int): The index of the previous city.
        
        Returns:
            float: The reward for visiting the current city.
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
            None
        
        Returns:
            None
        """
        pass