import torch as th

from controllers.controller import Controller
from learners.reinforce_learner import ReinforceLearner

class BiasedReinforceLearner (ReinforceLearner):
    def __init__(self, model, controller: Controller = None, params : dict = {}):
        super().__init__(model=model, controller=controller, params=params)
        self.value_criterion = th.nn.MSELoss()
        self.advantage_bias = params.get('advantage_bias', True)
        self.value_targets = params.get('value_targets', 'returns')
        self.gamma = params.get('gamma')
        self.compute_next_val = (self.value_targets == 'td')
    
    def _advantages(self, batch: dict, values: th.float32 = None, next_values: th.float32 = None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        advantages = batch['returns']
        if self.advantage_bias:
            advantages -= values
        return advantages
    
    def _value_loss(self, batch: dict , values: th.float32 = None, next_values: th.float32 = None):
        """ Computes the value loss (if there is one). """
        targets = None
        if self.value_targets == 'returns':
            targets = batch['returns']
        elif self.value_targets == 'td':
            targets = batch['rewards'] + self.gamma * (~batch['dones'] * next_values)
        return self.value_criterion(values, targets.detach())