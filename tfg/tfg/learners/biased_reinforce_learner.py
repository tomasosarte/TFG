import torch as th

from controllers.controller import Controller
from learners.reinforce_learner import ReinforceLearner

class BiasedReinforceLearner (ReinforceLearner):
    """
    This class is used to represent a biased REINFORCE learner.
    """
    def __init__(self, model: th.nn.Module, controller: Controller = None, params : dict = {}) -> None:
        """
        Initializes the biased REINFORCE learner extending the REINFORCE learner.

        Args:
            model th.nn.Module: The model used in the training process.
            controller Controller: The controller used in the training process.
            params dict: A dictionary containing the parameters for the learner.

        Returns:
            None
        """
        super().__init__(model=model, controller=controller, params=params)
        self.value_criterion = th.nn.MSELoss()
        self.advantage_bias = params.get('advantage_bias', True)
        self.value_targets = params.get('value_targets', 'returns')
        self.gamma = params.get('gamma', 0.99)
        self.compute_next_val = (self.value_targets == 'td')
    
    def _advantages(self, batch: dict, values: th.Tensor = None, next_values: th.Tensor = None) -> th.Tensor:
        """ 
        Computes the advantages, Q-values or returns for the policy loss. 
        
        Args:
            batch dict: A dictionary containing the batch of transitions.
            values th.Tensor: The values predicted by the model.
            next_values th.Tensor: The values predicted by the model for the next state.
            
        Returns:
            th.Tensor: A tensor containing the advantages, Q-values or returns.
        """
        advantages = batch['returns']
        if self.advantage_bias:
            advantages -= values
        return advantages
    
    def _value_loss(self, batch: dict , values: th.Tensor = None, next_values: th.Tensor = None) -> th.Tensor:
        """ 
        Computes the value loss (if there is one). 
        
        Args:
            batch dict: A dictionary containing the batch of transitions.
            values th.Tensor: The values predicted by the model.
            next_values th.Tensor: The values predicted by the model for the next state.
        
        Returns:
            th.Tensor: A tensor containing the value loss.
        """
        targets = None
        if self.value_targets == 'returns':
            targets = batch['returns']
        elif self.value_targets == 'td':
            targets = batch['rewards'] + self.gamma * (~batch['dones'] * next_values)
        return self.value_criterion(values, targets.detach())