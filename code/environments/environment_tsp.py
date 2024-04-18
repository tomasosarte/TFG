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

        # Init of Environment class
        super().__init__()

        # Check the shape of the cities tensor
        assert cities.shape[1] == node_dimension, "The cities tensor must have shape (n, 2)"

        # The number of cities in the TSP instance and the cities themselves
        self.n_cities = cities.shape[0]
        self.cities = cities 

        # Tensor indicating which cities have been visited with a 1 and which have not been visited with a 0
        visited_cities = th.zeros(self.n_cities)
        
        # Form symbol tensor
        symbol = th.ones(1, dtype=th.float32)*(-1)

        # Form the state tensor
        flat_cities = cities.flatten()

        if self.n_cities < max_nodes_per_graph:
            visited_cities = th.cat((visited_cities, th.ones(max_nodes_per_graph - self.n_cities)), dim=0)
            flat_cities = th.cat((flat_cities, th.zeros((max_nodes_per_graph - self.n_cities)*node_dimension)))

        self.state = th.cat((th.tensor([self.n_cities]), symbol, symbol, symbol, visited_cities, flat_cities)).unsqueeze(0)
        self.state_shape = (4+max_nodes_per_graph+max_nodes_per_graph*node_dimension,)
    
    def _reward(self, city1: int, city2: int):
        """
        Calculates the reward for moving from city1 to city2.
        
        Args:
            city1 int: The index of the first city.
            city2 int: The index of the second city.
        
        Returns:
            float: The reward for moving from city1 to city2.
        """
        # Calculate the distance between the two cities
        distance = th.norm(self.cities[city1] - self.cities[city2])
        return -distance
    
    def _get_state(self):
        """
        Returns the current state of the environment.
        
        Args:
            None
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
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
        self.state[0][1] = -1 # first city
        self.state[0][2] = -1 # current city
        self.state[0][3] = -1 # previous city
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
        if self.state[0][1] == -1:
            self.state[0][1] = action # first city
            self.state[0][2] = action # current city
            assert self.state[0][4+action] == 0, "The first city has already been visited"
            self.state[0][4+action] = 1
            reward = 0
            done = False
        else:
            self.state[0][3] = self.state[0][2] # previous city
            self.state[0][2] = action # current city
            assert self.state[0][4+action] == 0, "The city has already been visited"
            self.state[0][4+action] = 1
            reward = self._reward(self.state[0][2].type(th.int32), self.state[0][3].type(th.int32))
            done = (self.state[0][4:4+self.n_cities].sum() == self.n_cities)
        
        self.elapsed_time += 1
        next_state = self._get_state()
        return state, th.tensor([[reward]], dtype=th.float32), th.tensor([[done]]), next_state
