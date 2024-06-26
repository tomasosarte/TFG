import torch as th 

from controllers.controller import Controller

class ReinforceLearner:

    def __init__(self, model: th.nn.Module, controller: Controller = None, params: dict = {}) -> None:
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
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 0)
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.all_parameters = list(model.parameters())
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.compute_next_val = False  # whether the next state's value is computed
        self.old_pi = None  # this variable can be used for your PPO implementation
        self.min_pi = params.get('min_pi', 1E-6) # minimum probability for the policy in order to avoid NaNs
                                

        # Entropy regularization
        self.entropy_weight = params.get('entropy_weight', 0.01)
        self.entropy_regularization = params.get('entropy_regularization', False)
        self.decay_entropy = params.get('decay_entropy', False)
        self.entropy_weight_start = params.get('entropy_weight_start', 0.1)
        self.entropy_weight_end = params.get('entropy_weight_end', 0.01)
        self.entropy_anneal_time = params.get('entropy_anneal_time', 1E6)
        self.episode = 0

    def set_controller(self, controller: Controller) -> None:
        """
        This function is called in the experiment to set the controller

        Args: 
            Controller: Controller object that is used to generate the next state.

        Returns:
            None
        """
        self.controller = controller

    def _advantages(self, batch: dict, values: th.float32 = None, next_values: th.float32 = None) -> th.Tensor:
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
    
    def _value_loss(self, batch: dict, values: th.float32 = None, next_values: th.float32 = None) -> th.Tensor:
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
    
    def _policy_loss(self, pi: th.Tensor, advantages : th.Tensor) -> th.Tensor:
        """ 
        Computes the policy loss. 

        Args:
            pi: Policy.
            advantages: Advantages, Q-values or returns for the policy loss.
        
        Returns:
            Policy loss.
        """
        return -(advantages.detach() * pi.log()).mean()

    def _new_entropy_weight(self) -> None:
        """
        Updates the entropy weight.
        
        Args:
            None
        
        Returns:
            None
        """
        decay_factor = max(1 - self.episode / (self.entropy_anneal_time - 1), 0)
        self.entropy_weight = decay_factor * (self.entropy_weight_start - self.entropy_weight_end) + self.entropy_weight_end

    def train(self, batch: dict, episode: int) -> float:
        """
        Trains the model using the batch of data.

        Args:
            batch: Batch of data.
            episode: Episode number.
        
        Returns:
            loss_sum: Sum of the loss.
        """
        self.episode = episode
        if self.decay_entropy: self._new_entropy_weight()
        assert self.controller is not None, "Before train() is called, a controller must be specified. "
        # Set the model to training mode and old policy to None.
        self.model.train(True)
        self.old_pi, loss_sum = None, 0.0
        for _ in range(1 + self.offpolicy_iterations):
            # Compute the model-output for given batch
            out = self.model(batch['states'])   # compute both policy and values
            val = out[:, -1].unsqueeze(dim=-1)  # last entry are the values
            next_val = self.model(batch['next_states'])[:, -1].unsqueeze(dim=-1) if self.compute_next_val else None
            pi = self.controller.probabilities(state=batch['states'], out=out[:, :-1]).gather(dim=-1, index=batch['actions'])
            # pi = th.clamp(pi, min=self.min_pi)  # avoid NaNs

            # Combine policy and value loss
            policy_loss = self._policy_loss(pi, self._advantages(batch, val, next_val))
            value_loss = self.value_loss_param * self._value_loss(batch, val, next_val)
            loss = policy_loss + value_loss

            # Add entropy regularization
            if self.entropy_regularization: 
                entropy_loss = -(pi * pi.log()).sum(dim=1).mean()
                loss -= self.entropy_weight * entropy_loss
                
            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
            self.optimizer.step()
            loss_sum += loss.item()
    
        return loss_sum
