import torch as th
from environments.environment import Environment

class EnviornmentTSP(Environment):
    """
    This class is used to represent the enviornment of a TSP instance.
    """

    def __init__(self, cities: th.Tensor, params: dict = {}):
        """
        Constructor for the TSP environment class.
        
        Args:
            cities th.Tensor: A tensor representing the cities in the TSP instance.
            node_dimension int: The dimension of the node features.
            max_nodes_per_graph int: The maximum number of nodes that a graph can have.
        
        Returns:
            None
        """
        super().__init__()
        node_dimension = params.get('node_dimension', 2)
        max_nodes_per_graph = params.get('max_nodes_per_graph', 20)
        assert cities.shape[1] == node_dimension, "The cities tensor must have shape (n, 2)"

        # Init_vars
        self.n_cities = cities.shape[0]
        self.cities = cities 
        visited_cities = th.zeros(self.n_cities)
        symbol = th.ones(1, dtype=th.float32)*(-1)
        flat_cities = cities.flatten()
        self._max_episode_steps = self.n_cities + 1

        # Distance matrix
        self.distance_matrix = th.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.distance_matrix[i][j] = th.norm(self.cities[i] - self.cities[j])

        # Padding
        if self.n_cities < max_nodes_per_graph:
            padding_visited_cities = th.ones(max_nodes_per_graph - self.n_cities)
            visited_cities = th.cat((visited_cities, padding_visited_cities), dim=0)
            padding_cities = th.ones((max_nodes_per_graph - self.n_cities)*node_dimension)*(-1)
            flat_cities = th.cat((flat_cities, padding_cities))

        # State
        first_city, current_city, previous_city = symbol, symbol, symbol
        self.state = th.cat((th.tensor([self.n_cities]), first_city, current_city, previous_city, visited_cities, flat_cities)).unsqueeze(0)
        self.state_shape = (4 + max_nodes_per_graph + max_nodes_per_graph*node_dimension,)
    
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
        distance = self.distance_matrix[city1][city2]
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
        # Save current state
        state = self._get_state() 
        if self.state[0][1] == -1:
            assert self.state[0][4+action] == 0, "The first city has already been visited"
            self.state[0][1] = action # first city == action
            self.state[0][2] = action # current city == action
            self.state[0][4+action], reward, done = 1, 0, False
        else:
            # Determine if the episode is done with all cities visited
            all_visited = th.sum(self.state[0][4:4+self.n_cities]) == self.n_cities
            if all_visited:
                assert action == self.state[0][1], "The action must be the first city to complete the tour"  
                done = True              
            else: 
                assert self.state[0][4+action] == 0, "The first city has already been visited"
                self.state[0][4+action], done = 1, False
            self.state[0][3] = self.state[0][2] # previous city == current city
            self.state[0][2] = action # current city == first city
            reward = self._reward(self.state[0][3].type(th.int32), self.state[0][2].type(th.int32))
                
        self.elapsed_time += 1
        next_state = self._get_state()
        return state, th.tensor([[reward]], dtype=th.float32), th.tensor([[done]]), next_state
