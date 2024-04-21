import torch as th

from controllers.controller import Controller
from learners.actor_critic_learner import ActorCriticLearner

class OffpolicyActorCriticLearner (ActorCriticLearner):
    def __init__(self, model, controller: Controller = None, params: dict = {}):
        super().__init__(model=model, controller=controller, params=params)
    
    def _policy_loss(self, pi: th.Tensor, advantages: th.Tensor):
        """ Computes the policy loss. """
        if self.old_pi is None:
            self.old_pi = pi  # remember on-policy probabilities for off-policy losses
            # Return the default on-policy loss
            return super()._policy_loss(pi, advantages)
        else:
            # The loss for off-policy data
            ratios = pi / self.old_pi.detach()
            return -(advantages.detach() * ratios).mean()