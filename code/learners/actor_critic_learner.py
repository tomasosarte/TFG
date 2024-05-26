import torch as th

from learners.biased_reinforce_learner import BiasedReinforceLearner
from controllers.controller import Controller

class ActorCriticLearner(BiasedReinforceLearner):
    """
    This class is used to represent an actor-critic learner basded on the biased REINFORCE learner.
    """
    def __init__(self, model: th.nn.Module, controller: Controller=None, params: dict={}) -> None:
        """
        Initializes the actor-critic learner extending the biased REINFORCE learner.

        Args:
            model th.nn.Module: The model used in the training process.
            controller Controller: The controller used in the training process.
            params dict: A dictionary containing the parameters for the learner.
        
        Returns:
            None
        """
        super().__init__(model=model, controller=controller, params=params)
        self.advantage_bootstrap = params.get('advantage_bootstrap', True)
        self.compute_next_val = self.compute_next_val or self.advantage_bootstrap
    
    def _advantages(self, batch: dict, values: th.Tensor=None, next_values: th.Tensor=None) -> th.Tensor:
        """ 
        Computes the advantages, Q-values or returns for the policy loss. 
        
        Args:
            batch dict: A dictionary containing the batch of transitions.
            values th.Tensor: The values predicted by the model.
            next_values th.Tensor: The values predicted by the model for the next state.

        Returns:
            th.Tensor: A tensor containing the advantages, Q-values or returns.
        """
        advantages = None
        if self.advantage_bootstrap: 
            advantages = batch['rewards'] + self.gamma * (~batch['dones'] * next_values)
        else:
            advantages = batch['returns']
        if self.advantage_bias: 
            advantages = advantages - values
        return advantages