import torch as th

from controllers.controller import Controller
from learners.off_policy_actor_critic_learner import OffpolicyActorCriticLearner

class PPOLearner(OffpolicyActorCriticLearner):
    def __init__(self, model, controller: Controller = None, params: dict = {}):
        super().__init__(model=model, controller=controller, params=params)
        self.ppo_clipping = params.get('ppo_clipping', False)
        self.ppo_clip_eps = params.get('ppo_clip_eps', 0.2)
    
    def _policy_loss(self, pi: th.Tensor , advantages: th.Tensor):
        """ Computes the policy loss. """
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