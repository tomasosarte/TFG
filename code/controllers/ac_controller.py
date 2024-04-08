import torch as th
from threading import Lock
from controllers.controller import Controller

from torch.nn.modules import Module

class ActorCriticController(Controller):
    """
    Translates model outputs into greedy actions overwritting the controller object.
    """

    def copy(self):
        """
        Shallow copy of this controller that does not copy the model.

        Args:
            None
        
        Returns:
            GreedyController: The copied controller.
        """
        return ActorCriticController(model=self.model)

    def choose_action(self, state: dict) -> th.Tensor:
        """
        Returns a sample from the model's output distribution.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            th.Tensor: The action to take.
        """
        return th.distributions.Categorical(probs=self.probabilities(state)).sample()
        
    