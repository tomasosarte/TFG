from torch.nn.modules import Module
from experiments.experiment import Experiment
from controllers.ac_controller import ActorCriticController 
from learners.reinforce_learner import ReinforceLearner
from runners.runner import Runner
from environments.environment import Environment
from utils.transition_batch import TransitionBatch
import numpy as np
from runners.multi_runner import MultiRunner

class ActorCriticExperiment (Experiment):
    """
    Performs online actor-critic training overwriting the Experiment object.
    """
    def __init__(self, params: dict, model: Module, env: Environment, learner = None, **kwargs) -> None:
        """
        Initialize the ActorCriticExperiment object overwriting the Experiment init.

        Args: 
            params: Dictionary containing the parameters for the experiment.
            model: Model used in the training process.
            learner: Learner object that is used to train the model.
            kwargs: Additional arguments.
        
        Returns:
            None
        """
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ActorCriticController(model)
        self.env = env
        self.runner = MultiRunner(self.controller, params=params) if params.get('multi_runner', True) \
                      else Runner(self.controller, params=params)
        self.learner = ReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)

    def run(self) -> None:
        """
        Overriding the run method to perform online actor-critic training.

        Args:
            None
        
        Returns:
            None
        """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.batch_size)
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        for episode in range(self.max_episodes):
            # Run the policy fot batch_size steps
            batch = self.runner.run(self.batch_size, transition_buffer)
            env_steps += batch['env_steps']
            if batch['episode_length'] is not None:
                self.env_steps.append(env_steps)
                self.episode_lengths.append(batch['episode_length'])
                self.episode_returns.append(batch['episode_reward'])
            # Make a gradient update step
            loss = self.learner.train(batch['buffer'])
            self.episode_losses.append(loss)
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps: break
            # Show intermediate results
            if self.print_dots:
                print('.', end='')
            if self.plot_frequency is not None and (episode + 1) % self.plot_frequency == 0 \
                                               and len(self.episode_losses) > 2:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Iteration %u, 100-epi-return %.4g +- %.3g, length %u, loss %g' % 
                          (len(self.episode_returns), np.mean(self.episode_returns[-100:]), 
                           np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]), 
                           np.mean(self.episode_losses[-100:])))
