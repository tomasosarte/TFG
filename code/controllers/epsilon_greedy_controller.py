import numpy as np
import torch as th

class EpsilonGreedyController:
    """
    A wrapper that makes any controller into an epsilon-greedy controller. 
    Keeps track of training steps to decay exploration automatically.
    """
    def __init__(self, controller, params: dict = {}, exploration_step:int = 1) -> None:
        self.controller = controller
        self.max_epsilon = params.get('epsilon_start', 1.0)
        self.min_epsilon = params.get('epsilon_finish', 0.05)
        self.anneal_time = int(params.get('epsilon_anneal_time', 10000) / exploration_step)
        self.num_decisions = 0

    def epsilon(self) -> float:
        """
        Returns current epsilon.

        Args:
            None

        Returns:
            float: current epsilon
        """

        return th.max(1 - self.num_decisions / (self.anneal_time - 1), 0) * (self.max_epsilon - self.min_epsilon) + self.min_epsilon

    def choose(self, state, increase_counter=True, **kwargs) -> th.Tensor:
        """ 
        Returns the (possibly random) actions the agent takes when faced with "observation".
        Decays epsilon only when increase_counter = True".   

        Args:
            state (dict): The state of the environment.
            increase_counter (bool): Whether to increase the counter of decisions.
            **kwargs: Additional arguments to pass to the controller.

        Returns:
            th.Tensor: The action to take.     
        """
        eps = self.epsilon()
        if increase_counter: self.num_decisions += 1
        if np.random.rand() < eps: 
            # Get not visited cities
            not_visited_cities = state['not_visited_cities'].nonzero(as_tuple=False)
            num_not_visited = not_visited_cities.shape[0]
            node_to_visit = th.randint(0, num_not_visited, (1, ), dtype=th.long)
            return not_visited_cities[node_to_visit]

        else: 
            return self.controller.choose(state, **kwargs)