import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
from environments.environment import Environment

class Experiment:
    """
    Abstract class of an experiment. Contains logging and plotting functionality.
    """

    def __init__(self, params: dict, model: th.nn.Module, env: Environment, **kwargs) -> None:
        """
        Initialize the Experiment object.

        Args:
            params: Dictionary containing the parameters for the experiment.
            model: Model used in the training process.
            kwargs: Additional arguments.
        
        Returns:
            None
        """
        self.params = params
        self.plot_frequency = params.get('plot_frequency', 100)
        self.plot_train_samples = params.get('plot_train_samples', True)
        self.debug_messages = params.get('debug_messages', False)
        self.print_dots = params.get('print_dots', False)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_losses  = []
        self.env_steps = []
        self.total_run_time = 0.0
        self.device = params.get('device', 'cpu')
        self.use_tqdm = params.get('use_tqdm', False)
        self.final_plot = params.get('final_plot', False)
        self.wandb = params.get('wandb', False)

    def plot_training(self, update=False) -> None:
        """ 
        Plots logged training results. Use "update=True" if the plot is continuously updated
        or use "update=False" if this is the final call (otherwise there will be double plotting). 
        
        Args:
            update bool: Whether to update the plot or not.
        
        Returns:
            None
        """ 
        # Smooth curves
        window = max(int(len(self.episode_returns) / 50), 10)
        if len(self.episode_losses) < window + 2: return
        returns = np.convolve(self.episode_returns, np.ones(window)/window, 'valid')
        lengths = np.convolve(self.episode_lengths, np.ones(window)/window, 'valid')
        losses = np.convolve(self.episode_losses, np.ones(window)/window, 'valid')
        env_steps = np.convolve(self.env_steps, np.ones(window)/window, 'valid')
        # Determine x-axis based on samples or episodes
        if self.plot_train_samples:
            x_returns = env_steps
            x_losses = env_steps[(len(env_steps) - len(losses)):]
        else:
            x_returns = [i + window for i in range(len(returns))]
            x_losses = [i + len(returns) - len(losses) + window for i in range(len(losses))]
        # Create plot
        colors = ['b', 'g', 'r']
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()
        # Plot the losses in the left subplot
        pl.subplot(1, 3, 1)
        pl.plot(env_steps, returns, colors[0])
        pl.xlabel('environment steps' if self.plot_train_samples else 'episodes')
        pl.ylabel('episode return')
        # Plot the episode lengths in the middle subplot
        ax = pl.subplot(1, 3, 2)
        ax.plot(env_steps, lengths, colors[0])
        ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
        ax.set_ylabel('episode length')
        # Plot the losses in the right subplot
        ax = pl.subplot(1, 3, 3)
        ax.plot(x_losses, losses, colors[0])
        ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
        ax.set_ylabel('loss')
        # dynamic plot update
        display.clear_output(wait=True)
        if update:
            display.display(pl.gcf())

        # Save the plot


    def close(self) -> None:
        """ 
        Frees all allocated runtime ressources, but allows to continue the experiment later. 
        Calling the run() method after close must be able to pick up the experiment where it was. 
        
        Args:
            None
        
        Returns:
            None
        """
        pass

    def plot_rollout(self) -> None:
        """
        Dynamically plots a rollout of the agent in the environment, displaying one action at a time,

        Args:
            None
        
        Returns:
            None
        """ 
        pass
    
    def run(self) -> None:
        """ 
        Starts (or continues) the experiment. 
        
        Args:
            None
        
        Returns:
            None
        """
        assert False, "You need to extend the Expeirment class and override the method run(). "