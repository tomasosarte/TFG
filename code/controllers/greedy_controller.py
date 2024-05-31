from threading import Lock
import torch as th

from controllers.controller import Controller

class GreedyController(Controller):
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
        super(GreedyController, self).__init__(model)
        self.lock = Lock()
        self.max_cities = params.get('max_nodes_per_graph', 20)

    def copy(self) -> 'GreedyController':
        """
        Shallow copy of this controller that does not copy the model

        Args:
            None
        
        Returns:
            GreedyController: The copied controller.
        """
        return GreedyController(model=self.model)
    
    def probabilities(self, state: th.Tensor, **kwargs) -> th.Tensor:
        """ 
        Returns the probabilities with which the agent would choose actions (here one-hot because greedy).

        Args:
            state (th.Tensor): The state of the agent.
            **kwargs: Additional arguments.

        Returns:
            th.Tensor: The probabilities of each action. 
        """
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
        return th.zeros(*mx.shape).scatter_(dim=-1, index=th.max(mx, dim=-1)[1].unsqueeze(dim=-1), src=th.ones(1, 1))

    def choose_action(self, state: dict) -> th.Tensor:
        """
        Returns the greedy actions the agent would choose when facing a state.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            th.Tensor: The action to take.
        """
        return th.max(self.probabilities(state), dim=-1)[1].unsqueeze(0)
    