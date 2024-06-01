import torch as th

from controllers.controller import Controller
from learners.off_policy_actor_critic_learner import OffpolicyActorCriticLearner

class PPOLearner(OffpolicyActorCriticLearner):
    """
    This class is used to represent a PPO learner based on the off-policy actor-critic learner.
    """
    def __init__(self, model, controller: Controller = None, params: dict = {}) -> None:
        """
        Initializes the PPO learner extending the off-policy actor-critic learner.

        Args:
            model th.nn.Module: The model used in the training process.
            controller Controller: The controller used in the training process.
            params dict: A dictionary containing the parameters for the learner.
        
        Returns:
            None
        """
        super().__init__(model=model, controller=controller, params=params)
        self.ppo_clipping = params.get('ppo_clipping', False)
        self.ppo_clip_eps = params.get('ppo_clip_eps', 0.2)
    
    def _policy_loss(self, pi: th.Tensor , advantages: th.Tensor) -> th.Tensor:
        """ 
        Computes the policy loss. 
        
        Args:
            pi th.Tensor: The policy probabilities.
            advantages th.Tensor: The advantages, Q-values or returns.
        
        Returns:
            th.Tensor: A tensor containing the policy loss.
        """
        if self.old_pi is None:
            # The loss for on-policy data does not change
            return super()._policy_loss(pi, advantages)
        else:
            # The loss for off-policy data
            ratios = pi / self.old_pi.detach()
            loss = advantages.detach() * ratios
            if self.ppo_clipping:
                # off-policy loss with PPO clipping
                ppo_loss = th.clamp(ratios, 1-self.ppo_clip_eps, 1+self.ppo_clip_eps) * advantages.detach()
                loss = th.min(loss, ppo_loss)

            return -loss.mean()