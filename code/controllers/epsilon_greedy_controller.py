import numpy as np
import torch as th
from controllers.controller import Controller

class EpsilonGreedyController:
    """
    A wrapper that makes any controller into an epsilon-greedy controller. 
    Keeps track of training steps to decay exploration automatically.
    """
    def __init__(self, controller: Controller, params: dict = {}, exploration_step: int = 1) -> None:
        self.controller = controller
        self.max_epsilon = th.tensor(params.get('epsilon_start', 1.0), dtype=th.float32)
        self.min_epsilon = th.tensor(params.get('epsilon_finish', 0.05), dtype=th.float32)
        self.anneal_time = th.tensor(params.get('epsilon_anneal_time', 10000) / exploration_step, dtype=th.float32)
        self.num_decisions = th.tensor(0, dtype=th.float32)

    def epsilon(self) -> float:
        """
        Returns current epsilon.

        Args:
            None

        Returns:
            float: current epsilon
        """
        # print all variables
        return th.max(1 - self.num_decisions / (self.anneal_time - 1), th.tensor(0, dtype=th.float32)) * (self.max_epsilon - self.min_epsilon) + self.min_epsilon

    def probabilities(self, state: th.Tensor, out: th.Tensor = None) -> th.Tensor:

        """ 
        Returns the probabilities with which the agent would choose actions. 
        """
        eps = self.epsilon()

        # probabilities from controller
        controller_probabilities = self.controller.probabilities(state, out)

        # Get num of controller_probabilities that are not 0
        num_possible_actions = (controller_probabilities > 0).sum(dim=-1).view(-1, 1)

        # Add epsilon to all actions that are possible
        eps_padding = eps*th.ones(controller_probabilities.size(0), 1)/num_possible_actions
        probs = eps_padding + (1 - eps) * controller_probabilities
        probs[probs == eps_padding] = 0.0
        return probs
        
    def choose_action(self, state, increase_counter=True, **kwargs):
        """ Returns the (possibly random) actions the agent takes when faced with "observation".
            Decays epsilon only when increase_counter=True". """
        if increase_counter: self.num_decisions += 1
        return th.distributions.Categorical(probs=self.probabilities(state)).sample().unsqueeze(0)
    


