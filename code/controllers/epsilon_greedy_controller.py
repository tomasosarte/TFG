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

        # -------------- Environment tsp --------------
        self.max_cities = params.get('max_nodes_per_graph', 20)

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

    def probabilities(self, state: th.Tensor, **kwargs) -> th.Tensor:

        """ 
        Returns the probabilities with which the agent would choose actions. 
        """
        eps = self.epsilon()
        controller_probabilities = self.controller.probabilities(state, **kwargs)

        # Get visited cities
        visited_cities = state[0][4:4+self.max_cities].view(-1).type(th.bool)

        # Get num of possible actions
        num_possible_actions = (visited_cities == False).sum()

        if num_possible_actions == 0:
            first_city = state[0][1]
            assert first_city != -1, "First city is -1"
            probs = th.zeros(controller_probabilities.shape).scatter_(dim=-1, index=first_city.type(th.int64).view(1, 1), src=th.tensor([[1.0]]))
        else:
            # Get probs and set to 0 probs where controller_probabilities is 0
            probs = eps*th.ones(1, 1)/num_possible_actions + (1 - eps) * controller_probabilities
            idx_visited_cities = visited_cities.nonzero()
            probs[0][idx_visited_cities] = 0
            
        return probs
        
    def choose_action(self, state, increase_counter=True, **kwargs):
        """ Returns the (possibly random) actions the agent takes when faced with "observation".
            Decays epsilon only when increase_counter=True". """
        if increase_counter: self.num_decisions += 1
        return th.distributions.Categorical(probs=self.probabilities(state)).sample().unsqueeze(0)
