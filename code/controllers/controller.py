import torch as th
from threading import Lock

from torch.nn.modules import Module

class Controller:
    """
    Abstract class of a controller .
    """

    def __init__(self, model: th.nn.Module) -> None:
        """
        Initializes the controller.

        Args:
            model (th.nn.Module): The model to be used.
        
        Returns:
            None
        """
        self.lock = Lock()
        self.model = model

    def copy(self):
        """
        Shallow copy of the controller.

        Args:
            None
        
        Returns:
            Controller: The copied controller.
        """
        pass
    
    def parameters(self):
        """
        Returns a generator of the underlying model parameters

        Args:
            None
        
        Returns:
            generator: The parameters of the model.
        """
        return self.model.parameters()

    def choose_action(self, state: dict) -> th.Tensor:
        """
        Holder for choosing an action.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            None
        """
        pass

    def probabilities(self, state: th.Tensor) -> th.Tensor:
        """
        Probabilities of the model

        Args:
            state (th.Tensor): The state of the environment.
        
        Returns:
        """
        self.lock.acquire()
        try: probabilities, value = self.model(state)
        finally: self.lock.release()
        return probabilities
    