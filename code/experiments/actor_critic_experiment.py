from torch.nn.modules import Module
from controllers.epsilon_greedy_controller import EpsilonGreedyController
from experiments.experiment import Experiment
from controllers.ac_controller import ActorCriticController 
from learners.reinforce_learner import ReinforceLearner
from runners.runner import Runner
from environments.environment import Environment
from utils.transition_batch import TransitionBatch
import numpy as np
from runners.multi_runner import MultiRunner
import matplotlib.pyplot as plt
import torch as th
from IPython.display import display, clear_output
import pylab as pl
from IPython import display

class ActorCriticExperiment(Experiment):
    """
    Performs online actor-critic training overwriting the Experiment object.
    """
    def __init__(self, params: dict, model: Module, env: Environment, learner: ReinforceLearner, **kwargs) -> None:
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
        super().__init__(params, model, env, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ActorCriticController(model, params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.env = env
        self.runner = MultiRunner(self.controller, env=env, params=params) if params.get('multi_runner', True) \
                      else Runner(self.controller, env=env, params=params)
        self.learner = learner
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
            if self.plot_frequency is not None and (episode + 1) % self.plot_frequency == 0 and len(self.episode_losses) > 2:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Iteration %u, 100-epi-return %.4g +- %.3g, length %u, loss %g' % 
                          (len(self.episode_returns), np.mean(self.episode_returns[-100:]), 
                           np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]), 
                           np.mean(self.episode_losses[-100:])))
                    
        return np.mean(self.episode_returns[-100:])
    
    def plot_rollout(self) -> None:
        """
        Overriding the plot_rollout method to plot the rollout of the agent.

        Args:
            None
        
        Returns:
            None
        """
        batch = self.runner.run_episode()
        states = batch['buffer']['states'].cpu()
        actions = batch['buffer']['actions'].cpu()
        cities = self.env.cities.cpu()

        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_label('Rollout')
        ax.set_xlabel('Coord X')
        ax.set_ylabel('Coord Y')

        # Draw the cities
        ax.plot(cities[:, 0], cities[:, 1], 'o', color='black')
        
        for i in range(1, states.shape[0]):
            current_city = cities[states[i][2].type(th.int32)]
            next_city = cities[actions[i].squeeze(0).type(th.int32)]

            # Draw the arrow
            ax.arrow(current_city[0], current_city[1], next_city[0] - current_city[0], next_city[1] - current_city[1], 
                     head_width=0.03, head_length=0.02, fc='blue', ec='blue')

            # Sleep for a short while to slow down the animation
            display.clear_output(wait=True)
            display.display(pl.gcf())

            
            