import torch as th

from controllers.controller import Controller

class GreedyController(Controller):
    """
    Translates model outputs into greedy actions overwritting the controller object.
    """

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
        self.lock.acquire()
        try: output = self.model(state)
        finally: self.lock.release()
        return th.zeros(*output[:, :-1].shape).scatter_(dim=-1, index=th.max(output[:, :-1], dim=-1)[1].unsqueeze(dim=-1), src=th.ones(1, 1))

    def choose_action(self, state: dict) -> th.Tensor:
        """
        Returns the greedy actions the agent would choose when facing a state.

        Args:
            state (dict): The state of the environment.
        
        Returns:
            th.Tensor: The action to take.
        """
        return th.max(self.probabilities(state), dim=-1)[1].unsqueeze(0)
    