import math
import torch as th
from controllers.controller import Controller

class EpsilonGreedyController:
    """
    A wrapper that makes any controller into an epsilon-greedy controller. 
    Keeps track of training steps to decay exploration automatically.
    """
    def __init__(self, controller: Controller, params: dict = {}, exploration_step: int = 1) -> None:
        self.controller = controller
        self.max_epsilon = th.tensor(params.get('epsilon_start', 1.0), dtype=th.float16)
        self.min_epsilon = th.tensor(params.get('epsilon_finish', 0.05), dtype=th.float16)
        self.anneal_time = th.tensor(params.get('epsilon_anneal_time', 10000) / exploration_step, dtype=th.float32)
        self.decay_type = params.get('epsilon_decay', 'linear')
        self.num_decisions = th.tensor(0, dtype=th.float32)
        self.device = params.get('device', 'cpu')
        self.zero = th.tensor(0, dtype=th.float16)

    def epsilon(self) -> float:
        """
        Returns current epsilon.

        Args:
            None

        Returns:
            float: current epsilon
        """
        # print all variables
        if self.decay_type == "linear": 
            decay_factor = max(1 - self.num_decisions / (self.anneal_time - 1), 0)
            return decay_factor * (self.max_epsilon - self.min_epsilon) + self.min_epsilon
        elif self.decay_type == "exponential": 
            decay_factor = -math.log(self.min_epsilon / self.max_epsilon) / (self.anneal_time - 1)
            return self.max_epsilon * math.exp(-decay_factor * self.num_decisions)
        

    def probabilities(self, state: th.Tensor, out: th.Tensor = None) -> th.Tensor:

        """ 
        Returns the probabilities with which the agent would choose actions. 
        """
        eps = self.epsilon()

        # probabilities from controller
        controller_probabilities = self.controller.probabilities(state, out)

        # Get num of controller_probabilities that are not 0
        num_possible_actions = (controller_probabilities > 0).sum(dim=-1, keepdim=True)

        # Get idx of controller_probabilities that are equal to 0
        mask_zero = (controller_probabilities == 0)

        # Add epsilon to all actions that are possible
        eps_padding = eps*th.ones(controller_probabilities.size(0), 1)/num_possible_actions
        probs = eps_padding + (1 - eps) * controller_probabilities
        # probs[probs == eps_padding] = 0.0

        # Put to 0 all idx_zero
        probs[mask_zero] = 0.0

        return probs
        
    def choose_action(self, state, increase_counter=True, **kwargs):
        """ Returns the (possibly random) actions the agent takes when faced with "observation".
            Decays epsilon only when increase_counter=True". """
        if increase_counter: self.num_decisions += 1
        return th.distributions.Categorical(probs=self.probabilities(state)).sample().unsqueeze(0)
    


