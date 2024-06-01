import torch as th

from environments.environment import Environment
from generators.tsp_generator import TSPGenerator

class EnviornmentTSP(Environment):
    """
    This class is used to represent the enviornment of a TSP instance.
    """

    def __init__(self, params: dict = {}) -> None:
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
        self.params = params
        self.symbol = th.ones(1, dtype=th.float32)*(-1)
        self.node_dimension = 2
        self.max_nodes_per_graph = params.get('max_nodes_per_graph', 10)
        self.diff_sizes = params.get('diff_sizes', False)
        self.num_train_instance_per_size = params.get('num_train_instance_per_size', 10)
        self.training_sizes = params.get('training_sizes', [self.max_nodes_per_graph])
        self.use_training_set = params.get('use_training_set', True)
        self.diff_cities = params.get('diff_cities', False)       
        self.train_generator = TSPGenerator()
        self.cities = self._get_new_cities()
        self._form_state()
        self.distance_matrix = self._get_distance_matrix()
    
    def copy(self):
        """
        Copies the environment.

        Args:
            None
        
        Returns:
            EnvironmentTSP: A copy of the environment.
        """
        return EnviornmentTSP(self.params)
    
    def _get_new_cities(self) -> th.Tensor:  
        """
        Generates a new set of cities for the TSP instance or loads a set from the training set
        depending on the configuration.

        Args: 
            None
        
        Returns:
            th.Tensor: A tensor representing the cities in the TSP instance.
        """   
        if self.diff_cities:
            size = self.max_nodes_per_graph
            if self.diff_sizes: 
                size = self.training_sizes[th.randint(0, len(self.training_sizes), (1,)).item()]    
            if self.use_training_set:
                instance = th.randint(0, self.num_train_instance_per_size, (1,)).item()
                return th.load(f"training/tsp/size_{size}/instance_{instance}.pt").clone() 
            else: return self.train_generator.generate_instance(size)
        else: return self.params.get('cities', None).clone()
                 
    def _get_distance_matrix(self) -> th.Tensor:
        """
        Returns the distance matrix for the cities in the TSP instance.

        Args:
            None
        
        Returns:
            th.Tensor: A tensor representing the distance matrix for the cities in the TSP instance.
        """
        distance_matrix = th.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                distance_matrix[i][j] = th.norm(self.cities[i] - self.cities[j])
        return distance_matrix

    def _form_state(self) -> th.Tensor:
        """
        Forms the state of the environment.

        Args:
            None

        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        # Init_vars
        self.n_cities = self.cities.shape[0]
        visited_cities = th.zeros(self.n_cities)
        
        flat_cities = self.cities.flatten()
        self._max_episode_steps = self.n_cities + 1

        # Padding
        if self.n_cities < self.max_nodes_per_graph:
            padding_visited_cities = th.ones(self.max_nodes_per_graph - self.n_cities)
            visited_cities = th.cat((visited_cities, padding_visited_cities), dim=0)
            padding_cities = th.ones((self.max_nodes_per_graph - self.n_cities)*self.node_dimension)*(-1)
            flat_cities = th.cat((flat_cities, padding_cities))

        # State
        first_city, current_city, previous_city = self.symbol, self.symbol, self.symbol
        self.state = th.cat((th.tensor([self.n_cities]), first_city, current_city, previous_city, visited_cities, flat_cities)).unsqueeze(0)
        self.state_shape = (4 + self.max_nodes_per_graph + self.max_nodes_per_graph*self.node_dimension,)
    
    def _reward(self, city1: int, city2: int) -> float:
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
    
    def _get_state(self) -> th.Tensor:
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

        if self.diff_cities:
            self.cities = self._get_new_cities()
            self._form_state()
            self.distance_matrix = self._get_distance_matrix()
        else:
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
        self.lock.acquire()
        try:
            # print("Action: ", action)
            state = self._get_state() 
            # print("State visited cities: ", self.state[0][4:4+self.n_cities])
            if self.state[0][1] == -1:
                assert self.state[0][4+action] == 0, "The city has already been visited"
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
                    assert self.state[0][4+action] == 0, "The city has already been visited"
                    self.state[0][4+action], done = 1, False
                self.state[0][3] = self.state[0][2] # previous city == current city
                self.state[0][2] = action # current city == first city
                reward = self._reward(self.state[0][3].type(th.int32), self.state[0][2].type(th.int32))
                    
            self.elapsed_time += 1
            next_state = self._get_state()
        finally: self.lock.release()
        return state, th.tensor([[reward]], dtype=th.float32), th.tensor([[done]]), next_state
