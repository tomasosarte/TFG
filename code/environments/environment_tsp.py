import torch as th
from environments.environment import Environment

class EnviornmentTSP(Environment):
    """
    This class is used to represent the enviornment of a TSP instance.
    """

    def __init__(self, cities: th.Tensor, node_dimension: int = 2, max_nodes_per_graph: int = 10):
        """
        Constructor for the TSP environment class.
        
        Args:
            cities th.Tensor: A list of City objects representing the cities in the TSP instance. The cities
                                are represented as a 2D tensor with shape (n, 2) where n is the number of cities.
            first_city th.Tensor: The first city that the agent visited.
        
        Returns:
            None
        """
        # Check the shape of the cities tensor
        assert cities.shape[1] == node_dimension, "The cities tensor must have shape (n, 2)"

        # Init of Environment class
        super().__init__()

        # The number of cities in the TSP instance and the cities themselves
        self.n_cities = cities.shape[0]
        self.cities = cities 

        # The current and previous city that the agent is at and the first city that the agent visited
        # self.current_city = th.tensor([-1], dtype=th.int32)
        # self.first_city = th.tensor([-1], dtype=th.int32)
        # self.previous_city = th.tensor([-1], dtype=th.int32)

        # Tensor of booleans indicating which cities have not been visited with a 1 and which have been visited with a 0 extended to the maximum number of nodes per graph
        # self.visited_cities = th.zeros([1, self.n_cities], dtype=th.bool)
        visited_cities = th.zeros(self.n_cities)
        
        # Form state
        symbol = th.ones(1, dtype=th.float32)*(-1)
        flat_cities = cities.flatten()
        if self.n_cities < max_nodes_per_graph:
            visited_cities = th.cat((visited_cities, th.ones(max_nodes_per_graph - self.n_cities)), dim=0)
            flat_cities = th.cat((flat_cities, th.zeros((max_nodes_per_graph - self.n_cities)*node_dimension)))
        self.state = th.cat((th.tensor([self.n_cities]), symbol, symbol, symbol, visited_cities, flat_cities)).unsqueeze(0)
        self.state_shape = (4+max_nodes_per_graph+max_nodes_per_graph*node_dimension,)

    def _distance(self, city1: int, city2: int):
        """
        Calculates the distance between two cities.
        
        Args:
            city1 int: The index of the first city.
            city2 int: The index of the second city.
        
        Returns:
            float: The distance between city1 and city2.
        """
        return th.norm(self.cities[city1] - self.cities[city2].type(th.int32))
    
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
            th.Tensor: A tensor representing the state of the environment.
        """
        # return {
        #      'current_cities': self.current_city,
        #      'first_cities': self.first_city,
        #      'previous_cities': self.previous_city,
        #      'not_visited_cities': self.not_visited_cities,
        #      'cities': self.cities,
        # }
        return self.state.clone()
    
    def reset(self) -> dict:
        """
        Resets the environment to the starting state.

        Args:
            None
        
        Returns:
            th.Tensor: The state of the environment after resetting.
        """
        super().reset()
        # self.current_city = th.tensor([-1], dtype=th.int32)
        # self.first_city = th.tensor([-1], dtype=th.int32)
        # self.previous_city = th.tensor([-1], dtype=th.int32)
        # self.not_visited_cities = th.ones(self.n_cities, dtype=th.bool).unsqueeze(0)
        self.state[0][1] = -1
        self.state[0][2] = -1
        self.state[0][3] = -1
        self.state[0][4:4+self.n_cities] = th.zeros(self.n_cities)
        return self._get_state()
        
    def step(self, action: float) -> tuple:
        """
        Takes a step in the environment by visiting the city at the given index.
        
        Args:
            action float: The index of the city to visit.
        
        Returns:
            state th.Tensor: The state of the environment after taking the step.
            reward float: The reward for taking the step.
            done bool: A boolean indicating if the episode is done.
            info dict: A dictionary of additional information.
        """
        
        state = self._get_state()
        # if self.first_city == -1:
        if self.state[0][1] == -1:
            # self.first_city[0] = action
            # self.current_city[0] = action
            # self.not_visited_cities[0][action] = False
            self.state[0][1] = action 
            self.state[0][2] = action
            self.state[0][4+action] = 1
            reward = 0
            done = False
        else:
            self.state[0][3] = self.state[0][2]
            self.state[0][2] = action
            self.state[0][4+action] = 1
            reward = self._reward(self.state[0][2].type(th.int32), self.state[0][3].type(th.int32))
            done = self.state[0][4:4+self.n_cities].sum() == self.n_cities
        
        self.elapsed_time += 1
        next_state = self._get_state()
        return state, th.tensor([[reward]], dtype=th.float32), th.tensor([[done]]), next_state
