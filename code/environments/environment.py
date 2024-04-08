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
        pass

    def reset(self) -> dict:
        """
        Resets the environment to the starting state.

        Args:
            None
        
        Returns:
            None
        """
        pass 
        
    def step(self, action: int):
        """
        Takes a step in the environment by visiting the city at the given index.
        
        Args:
            None
        
        Returns:
            None
        """
        
        pass