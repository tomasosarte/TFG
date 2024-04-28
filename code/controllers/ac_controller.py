import torch as th
from threading import Lock
from controllers.controller import Controller

from torch.nn.modules import Module

class ActorCriticController(Controller):
    """
    Translates model outputs into greedy actions overwritting the controller object.
    """

    def __init__(self, model: th.nn.Module, params: dict = {}) -> None:
        """
        Initializes the controller.

        Args:
            model (th.nn.Module): The model to be used.
        
        Returns:
            None
        """
        super(ActorCriticController, self).__init__(model)
        self.lock = Lock()
        self.max_cities = params.get('max_nodes_per_graph', 20)

    def copy(self):
        """
        Shallow copy of this controller that does not copy the model.

        Args:
            None
        
        Returns:
            GreedyController: The copied controller.
        """
        return ActorCriticController(model=self.model)

    def probabilities(self, state: th.Tensor, out: th.Tensor = None) -> th.Tensor:
        """
        Probabilities of the model

        Args:
            
        Returns:
        """
        self.lock.acquire()
        try: 
            mx = out if out != None else self.model(state)[:, :-1]
            visited_cities = state[:, 4:4+self.max_cities].type(th.bool)
            num_possible_actions = (~visited_cities).sum(dim=1).view(-1, 1)

            # Mask determining last states
            mask_last_states = (num_possible_actions == 0).view(-1, 1)
            if mask_last_states.any():
                idx_last_states = mask_last_states.view(-1).nonzero()

                # Get first cities of last states            
                first_cities = state[:, 1].type(th.int64).view(-1, 1)
                selected_first_cities = first_cities[idx_last_states].view(-1, 1)

                visited_cities[idx_last_states] = visited_cities[idx_last_states].view(-1, visited_cities.size(1)).\
                    scatter_(dim=-1, index=selected_first_cities, src=th.zeros(selected_first_cities.size(0),1).type(th.bool)).view(*visited_cities[idx_last_states].shape)

            infty_mask = visited_cities.type(th.float32) * float('-inf')
            infty_mask[th.isnan(infty_mask)] = 0.0
            mx = mx + infty_mask
        finally: self.lock.release()
        return th.nn.functional.softmax(mx, dim=-1)
    
    def choose_action(self, state: dict) -> th.Tensor:
        """
        Returns a sample from the model's output distribution.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            th.Tensor: The action to take.
        """
        return th.distributions.Categorical(probs=self.probabilities(state)).sample().unsqueeze(0)
        
    