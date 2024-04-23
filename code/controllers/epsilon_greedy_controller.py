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
        visited_city_tensor = state[0][4:4+self.max_cities].view(-1).type(th.bool)

        # Get num of possible actions
        num_possible_actions = (visited_city_tensor == False).sum()     
        if num_possible_actions == 0: num_possible_actions = 1

        # Get probs and set to 0 probs where controller_probabilities is 0
        probs = eps*th.ones(1, 1)/num_possible_actions + (1-eps) * controller_probabilities
        
        print(f"Probs: {probs}")
        print(f"Visited city tensor: {visited_city_tensor}")
        # Put 0 where the cities are visited
        probs[visited_city_tensor.view(1, -1)] = 0

        # If probs are all 0, set to 1 the first City
        if (visited_city_tensor == False).sum() == 0:
            first_city = state[0][1].item()
            print(f"First city: {first_city}")
            probs= th.zeros(probs.shape)
            probs[0][int(first_city)] = 1

        print(f"Probs after: {probs}")
        print('-'*50)

        return probs
    
    def choose_action(self, state, increase_counter=True, **kwargs):
        """ Returns the (possibly random) actions the agent takes when faced with "observation".
            Decays epsilon only when increase_counter=True". """
        if increase_counter: self.num_decisions += 1
        # print(f"State: {state[0][4:4+self.max_cities]}")
        # print(f"Probabilities:\n {self.probabilities(state)}")
        return th.distributions.Categorical(probs=self.probabilities(state)).sample().unsqueeze(0)

    # def choose_action(self, state: th.Tensor, increase_counter=True, **kwargs) -> th.Tensor:
    #     """ 
    #     Returns the (possibly random) actions the agent takes when faced with "observation".
    #     Decays epsilon only when increase_counter = True".   

    #     Args:
    #         state (th.Tensor): The state of the environment.
    #         increase_counter (bool): Whether to increase the counter of decisions.
    #         **kwargs: Additional arguments to pass to the controller.

    #     Returns:
    #         th.Tensor: The action to take.     
    #     """
    #     eps = self.epsilon()
    #     # print(f"Epsilon: {eps}")
    #     if increase_counter: self.num_decisions += 1
    #     if np.random.rand() < eps: 
    #         # Get not visited cities
    #         visited_cities_tensor = state[0][4:4+self.max_cities].view(-1)

    #         # Get indices of not visited cities
    #         not_visited_cities_indexes = th.nonzero(visited_cities_tensor == 0).view(-1)

    #         # Choose a random city if there are any left to visit
    #         if not_visited_cities_indexes.shape[0] != 0:
    #             node_to_visit = th.randint(0, not_visited_cities_indexes.shape[0], (1,)).item()
    #             return not_visited_cities_indexes[node_to_visit].view(1, 1)
        
    #     return self.controller.choose_action(state, **kwargs)