import wandb
import torch as th
import numpy as np
import pylab as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.modules import Module
from IPython.display import display, clear_output
from IPython import display

from runners.runner import Runner
from solvers.gurobi_tsp import solve_tsp
from runners.multi_runner import MultiRunner
from experiments.experiment import Experiment
from environments.environment import Environment
from utils.transition_batch import TransitionBatch
from learners.reinforce_learner import ReinforceLearner
from controllers.ac_controller import ActorCriticController 
from controllers.epsilon_greedy_controller import EpsilonGreedyController

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
        episodes = tqdm(range(self.max_episodes)) if self.use_tqdm  else range(self.max_episodes)
        for episode in episodes:
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
            if self.wandb: 
                wandb.log({'episode_return': batch['episode_reward'], 'episode_length': batch['episode_length'], 'loss': loss})
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps: break
            # Show intermediate results
            if self.print_dots:
                print('.', end='')
            if self.plot_frequency is not None and (episode + 1) % self.plot_frequency == 0 and len(self.episode_losses) > 2:
                self.plot_training(update=True)
            if self.debug_messages and (episode + 1) % 10 == 0 and len(self.episode_losses) > 2:
                print('Iteration %u, 100-epi-return %.4g +- %.3g, length %u, loss %g' % 
                        (len(self.episode_returns), np.mean(self.episode_returns[-100:]), 
                        np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]), 
                        np.mean(self.episode_losses[-100:])))
        
        # Plot the final results
        if self.final_plot:
            self.plot_training(update=False)
        
        return self.episode_returns, self.episode_lengths, self.episode_losses, self.env_steps

    def plot_tour(self, cities: th.Tensor, tour: th.Tensor, num_cities: int) -> None:
        """
        Plot the tour of the agent.

        Args:
            cities: Tensor containing the coordinates of the cities.
            tour: Tensor containing the tour of the agent.
            num_cities: Number of cities in the problem.
        
        Returns:
            None
        """

        # Create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Rollout')
        ax.set_xlabel('Coord X')
        ax.set_ylabel('Coord Y')

        # Draw the cities
        ax.plot(cities[:, 0], cities[:, 1], 'o', color='black')
        
        for i in range(1, num_cities + 1):
            current_city = cities[tour[i-1].squeeze(0).type(th.int32)]
            next_city = cities[tour[i].squeeze(0).type(th.int32)]

            # Calculate the direction and length
            dx = next_city[0] - current_city[0]
            dy = next_city[1] - current_city[1]
            length = (dx**2 + dy**2)**0.5

            # Adjust the arrow to end at the next city
            arrow_length = length - 0.03  # slightly less than full length to stop at the point

            # Draw the arrow
            ax.arrow(current_city[0], current_city[1], 
                    arrow_length * (dx / length), arrow_length * (dy / length), 
                    head_width=0.03, head_length=0.02, fc='blue', ec='blue')

            # Sleep for a short while to slow down the animation
            display.clear_output(wait=True)
        display.display(plt.gcf())

    def plot_rollout(self) -> None:
        """
        Overriding the plot_rollout method to plot the rollout of the agent.

        Args:
            None
        
        Returns:
            None
        """
        
        # Run the episode
        batch = self.runner.run_episode()
        states = batch['buffer']['states'].cpu()
        actions = batch['buffer']['actions'].cpu()

        # Get the cities
        init_cities = 4 + self.runner.max_nodes_per_graph
        cities = states[0, init_cities:].reshape(-1, 2)
        num_cities = int(states[0][0].item())

        # Plot the tour
        self.plot_tour(cities, actions, num_cities)

    def calculate_tour_distance(self, cities: th.Tensor, tour: th.Tensor, num_cities: int) -> float:
        """
        Calculate the distance of the tour.

        Args:
            cities: Tensor containing the coordinates of the cities.
            tour: Tensor containing the tour of the agent.
            num_cities: Number of cities in the problem.
        
        Returns:
            float: The distance of the tour.
        """
        distance = 0.0
        for i in range(1, num_cities + 1):
            current_city = cities[tour[i-1].squeeze(0).type(th.int32)]
            next_city = cities[tour[i].squeeze(0).type(th.int32)]
            dx = next_city[0] - current_city[0]
            dy = next_city[1] - current_city[1]
            distance += (dx**2 + dy**2)**0.5
        return distance.item()
    
    def compare_vs_baseline(self, num_episodes: int = 10) -> float:
        """
        Compare the model's performance against the optimal solution.

        Args:
            num_episodes: Number of episodes to run.
        
        Returns:
            float: The average gap between the model's solution and the optimal solution.
        """

        total_gap = 0.0
        for _ in range(num_episodes):
            # Run the episode
            batch = self.runner.run_episode()
            states = batch['buffer']['states'].cpu()
            actions = batch['buffer']['actions'].cpu()

            # Get the cities
            num_cities = int(states[0][0].item())
            init_cities = 4 + self.runner.max_nodes_per_graph
            cities = states[0, init_cities:].reshape(-1, 2)
            cities = cities[:num_cities]

            # Calculate the optimal solution
            optim_tour, optim_distance = solve_tsp(cities)

            # Calculate the model's solution
            model_tour = actions[:num_cities + 1].squeeze(1).cpu()
            model_distance = self.calculate_tour_distance(cities, model_tour, num_cities)

            # Calculate the gap
            gap = round((model_distance - optim_distance) / optim_distance * 100, 2)
            total_gap += gap

        avg_gap = total_gap / num_episodes
        return avg_gap
            


            