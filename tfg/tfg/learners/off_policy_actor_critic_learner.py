import torch as th

from controllers.controller import Controller
from learners.actor_critic_learner import ActorCriticLearner

class OffpolicyActorCriticLearner(ActorCriticLearner):
    """
    This class is used to represent an off-policy actor-critic learner based on the actor-critic learner.
    """
    def __init__(self, model: th.nn.Module, controller: Controller = None, params: dict = {}) -> None:
        """
        Initializes the off-policy actor-critic learner extending the actor-critic learner.

        Args:
            model th.nn.Module: The model used in the training process.
            controller Controller: The controller used in the training process.
            params dict: A dictionary containing the parameters for the learner.
        
        Returns:
            None
        """
        super().__init__(model=model, controller=controller, params=params)
    
    def _policy_loss(self, pi: th.Tensor, advantages: th.Tensor) -> th.Tensor:
        """ 
        Computes the policy loss. 
        
        Args:
            pi th.Tensor: The policy probabilities.
            advantages th.Tensor: The advantages, Q-values or returns.
        
        Returns:
            th.Tensor: A tensor containing the policy loss.
        """
        if self.old_pi is None:
            self.old_pi = pi  # remember on-policy probabilities for off-policy losses
            # Return the default on-policy loss
            return super()._policy_loss(pi, advantages)
        else:
            # The loss for off-policy data
            ratios = pi / self.old_pi.detach()
            return -(advantages.detach() * ratios).mean()