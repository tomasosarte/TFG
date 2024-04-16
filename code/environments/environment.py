import torch as th

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
        self.elapsed_time = 0
        self.state_shape = None

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