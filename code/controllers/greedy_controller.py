import torch as th
from threading import Lock
from controllers.controller import Controller

from torch.nn.modules import Module

class GreedyController(Controller):
    """
    Translates model outputs into greedy actions overwritting the controller object.
    """

    def copy(self):
        """
        Shallow copy of this controller that does not copy the model

        Args:
            None
        
        Returns:
            GreedyController: The copied controller.
        """
        return GreedyController(model=self.model)

    def choose_action(self, state: dict) -> th.Tensor:
        """
        Returns the greedy actions the agent would choose when facing a state.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            th.Tensor: The action to take.
        """
        return th.max(self.probabilities(state), dim=-1)[1]
    