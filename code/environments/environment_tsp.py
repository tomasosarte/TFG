import torch as th
from environments.environment import Environment

class EnvironemntTSP(Environment):
    """
    This class is used to represent the enviornment of a TSP instance.
    """

    def __init__(self, cities: th.Tensor):
        """
        Constructor for the TSP environment class.
        
        Args:
            cities th.Tensor: A list of City objects representing the cities in the TSP instance. The cities
                                are represented as a 2D tensor with shape (n, 2) where n is the number of cities.
            first_city th.Tensor: The first city that the agent visited.
        
        Returns:
            None
        """
        assert cities.shape[1] == 2, "The cities tensor must have shape (n, 2)"
        # The number of cities in the TSP instance and the cities themselves
        self.n_cities = cities.shape[0]
        self.cities = cities
        # The current and previous city that the agent is at and the first city that the agent visited
        self.current_city = None
        self.first_city = None
        self.previous_city = None
        # Tensor of booleans indicating which cities have not been visited with a 1 and which have been visited with a 0
        self.not_visited_cities = th.ones(self.n_cities, dtype=th.bool)
        # Elapsed time in the environment
        self.elapsed_time = 0

    def _distance(self, city1: int, city2: int):
        """
        Calculates the distance between two cities.
        
        Args:
            city1 int: The index of the first city.
            city2 int: The index of the second city.
        
        Returns:
            float: The distance between city1 and city2.
        """
        return th.norm(self.cities[city1] - self.cities[city2])
    
    def _reward(self, city1: int, city2: int):
        """
        Calculates the reward for moving from city1 to city2.
        
        Args:
            city1 int: The index of the first city.
            city2 int: The index of the second city.
        
        Returns:
            float: The reward for moving from city1 to city2.
        """
        return -self._distance(city1, city2)
    
    def _get_state(self):
        """
        Returns the current state of the environment.
        
        Args:
            None
        
        Returns:
            dict: A dictionary representing the state of the environment.
        """
        return {
            'current_city': self.current_city,
            'first_city': self.first_city,
            'previous_city': self.previous_city,
            'not_visited_cities': self.not_visited_cities,
            'cities': self.cities,
        }

    def reset(self) -> dict:
        """
        Resets the environment to the starting state.

        Args:
            None
        
        Returns:
            th.Tensor: The state of the environment after resetting.
        """
        self.current_city = None
        self.first_city = None
        self.previous_city = None
        self.not_visited_cities = th.ones(self.n_cities, dtype=th.bool)
        return self._get_state()
        
    def step(self, action: int):
        """
        Takes a step in the environment by visiting the city at the given index.
        
        Args:
            action int: The index of the city to visit.
        
        Returns:
            state th.Tensor: The state of the environment after taking the step.
            reward float: The reward for taking the step.
            done bool: A boolean indicating if the episode is done.
            info dict: A dictionary of additional information.
        """
        
        state = self._get_state()
        if self.first_city is None:
            self.first_city = action
            self.current_city = action
            self.not_visited_cities[action] = False
            reward = 0
            done = False
        else:
            self.previous_city = self.current_city
            self.current_city = action
            self.not_visited_cities[action] = False
            reward = self._reward(self.current_city, self.previous_city)
            done = self.not_visited_cities.sum() == 0
        
        self.elapsed_time += 1
        next_state = self._get_state()
        return state, reward, done, next_state
