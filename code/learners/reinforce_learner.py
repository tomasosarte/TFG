import torch as th 

class ReinforceLearner:

    def __init__(self, model: th.nn.Module, controller = None, optimizer: th.optim.Optimizer = None, params: dict = {}) -> None:
        """
        Initialize the ReinforceLearner object.

        Args:
            model: Model used in the training process.
            controller: Controller object that is used to generate the next state.
            optimizer: Optimizer used in the training process.
            params: Dictionary containing the parameters for the training process.
        
        Returns:
            None
        """
        # Model used in the training process.
        self.model = model
        self.controller = controller
        self.learning_rate = params.get('lr', 0.001)
        # Optimizer used in the training process.
        if optimizer is None: self.optimizer = th.optim.Adam(model.parameters(), lr=self.learning_rate)
        else: self.optimizer = optimizer
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 0)
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.all_parameters = list(model.parameters())
        self.compute_next_val = False  # whether the next state's value is computed
        self.old_pi = None  # this variable can be used for your PPO implementation
        self.max_cities = params.get('max_cities', 20)

    def set_controller(self, controller) -> None:
        """
        This function is called in the experiment to set the controller

        Args: 
            Controller: Controller object that is used to generate the next state.

        Returns:
            None
        """
        self.controller = controller

    def _advantages(self, batch, values=None, next_values=None):
        """ 
        Computes the advantages, Q-values or returns for the policy loss.

        Args:
            batch: Batch of data.
            values: Values of the states.
            next_values: Values of the next states.
        
        Returns:
            Advantages, Q-values or returns for the policy loss.
        """
        return batch['returns']
    
    def _value_loss(self, batch, values=None, next_values=None):
        """
        Computes the value loss (if there is one). 
        
        Args:
            batch: Batch of data.
            values: Values of the states.
            next_values: Values of the next states.

        Returns:
            Value loss.
        """
        return 0
    
    def _policy_loss(self, pi, advantages):
        """ 
        Computes the policy loss. 

        Args:
            pi: Policy.
            advantages: Advantages, Q-values or returns for the policy loss.
        
        Returns:
            Policy loss.
        """
        return -(advantages.detach() * pi.log()).mean()

    def train(self, batch) -> None:
        """
        Trains the model using the batch of data.

        Args:
            batch: Batch of data.
        
        Returns:
            loss_sum: Sum of the loss.
        """
        assert self.controller is not None, "Before train() is called, a controller must be specified. "
        # Set the model to training mode and old policy to None.
        self.model.train(True)
        self.old_pi, loss_sum = None, 0.0
        for _ in range(1 + self.offpolicy_iterations):
            # Compute model output for the batch.
            policies, values = self.model(batch['states']) # Compute policy and values
            _, next_values = self.model(batch['next_states']) if self.compute_next_val else None, None
            # Combine policy and value loss
            loss = self._policy_loss(policies, self._advantages(batch, next_values)) + self.value_loss_param * self._value_loss(batch, values, next_values)
            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum



        