import torch as th
import torch.nn as nn

class BasicNetwork(nn.Module):

    def __init__(self,
                max_nodes_per_graph: int,
                node_dimension: int = 2,
                embedding_dimension: int = 4,
                ) -> None:
        """
        Constructor for the BasicNetwork class. This class is used to represent the network that will be used to solve NP-HARD problems.

        Args:
            max_nodes_per_graph (int): The maximum number of nodes that a graph can have.
            node_dimension (int): The dimension of the node features.
            embedding_dimension (int): The dimension of the node embeddings.
        
        Returns:
            None
        """
        super(BasicNetwork, self).__init__()

        self.max_nodes_per_graph = max_nodes_per_graph
        self.node_dimension = node_dimension
        self.embedding_dimension = embedding_dimension

        # Input symbol for first step the size of the node dimension
        self.symbol = th.ones(node_dimension, dtype=th.float32)*(-1)

        # To map input node features to the embedding space
        self.initial_embedding = nn.Linear(node_dimension, embedding_dimension)
        
        # Layers of the network
        self.layers = nn.Sequential(
            nn.Linear((max_nodes_per_graph + 2) * embedding_dimension, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, max_nodes_per_graph + 1)
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        """
        Forward pass of the network. This function takes a state th.Tensor as input and returns the output of the network.

        Args:
            state (th.Tensor): A tensor containing the state of the agent in the environment.

        Returns:
            th.Tensor: The output of the network.
        """
        # Squeeze the state tensor
        state = state.squeeze()
        print(state)
        print('-'*50)
        # Check the shape of the state tensor
        num_cities = state[0].type(th.int)
        assert num_cities <= self.max_nodes_per_graph, "The number of cities in the graph is greater than the maximum number of nodes per graph"
        assert state.shape[0] == (4 + self.max_nodes_per_graph + self.node_dimension*self.max_nodes_per_graph), "The state tensor has the wrong shape"

        current_city = state[1].type(th.int)
        first_city = state[2].type(th.int)
        previous_city = state[3].type(th.int)
        not_visited_cities = state[4:4+num_cities].type(th.bool)
        cities = state[4+self.max_nodes_per_graph:4+self.max_nodes_per_graph + self.node_dimension*num_cities].view(num_cities, self.node_dimension)

        # Get the embeddig of the first and the current city
        if first_city == -1:
            embedded_first_city = self.initial_embedding(self.symbol)
            embedded_current_city = self.initial_embedding(self.symbol)
        else:
            first_city_coords = cities[first_city]
            current_city_coords = cities[current_city]
            embedded_first_city = self.initial_embedding(first_city_coords)
            embedded_current_city = self.initial_embedding(current_city_coords)

        # Embedd the cities tensor
        embedded_cities = self.initial_embedding(cities)

        # Cat the embedded first city, embedded current city and the cities tensor
        input_tensor = th.cat((embedded_first_city.unsqueeze(0), embedded_current_city.unsqueeze(0), embedded_cities))

        # If n_cities < max_nodes_per_graph, pad the input tensor with zeros and the not visited cities tensor
        if num_cities < self.max_nodes_per_graph:
            input_tensor = th.cat((input_tensor, th.zeros(self.max_nodes_per_graph - num_cities, self.embedding_dimension)), dim=0)
            not_visited_cities = th.cat((not_visited_cities, th.zeros(self.max_nodes_per_graph - num_cities, dtype=th.bool)), dim=0)

        # Reshape the input tensor to be a 1D tensor
        input_tensor = input_tensor.view(-1)

        # Forward pass
        output = self.layers(input_tensor)

        # Pass the mask that represents not visited cities to the output tensor. Giving them 0 probaility of being selected
        output_masked = output[:-1].masked_fill(~not_visited_cities, -float('inf'))

        # Transform the output into probabilities
        output_probabilities = th.softmax(output_masked, dim=0)

        # Get value
        value = output[-1]

        return output_probabilities, value
    
    # def forward(self, state: dict) -> th.Tensor:
    #     """
    #     Forward pass of the network. This function takes a state dictionary as input and returns the output of the network.

    #     Args:
    #         state (dict): A dictionary containing the state of the agent in the environment.

    #     Returns:
    #         th.Tensor: The output of the network.
    #     """
    #     # Get the embeddin of the first and the current city
    #     if state['first_city'] == None:
    #         embedded_first_city = self.initial_embedding(self.symbol)
    #         embedded_current_city = self.initial_embedding(self.symbol)
    #     else:
    #         first_city = state['cities'][state['first_city'].type(th.int)]
    #         current_city = state['cities'][state['current_city'].type(th.int)]
    #         embedded_first_city = self.initial_embedding(first_city)
    #         embedded_current_city = self.initial_embedding(current_city)

    #     # Check how many cities there are in the current graph and the node dimension
    #     n_cities = state['cities'].shape[0]
    #     node_dimension = state['cities'].shape[1]

    #     assert n_cities <= self.max_nodes_per_graph, "The number of cities in the graph is greater than the maximum number of nodes per graph"
    #     assert node_dimension == self.node_dimension, "The node dimension of the cities tensor is different from the node dimension of the network"

    #     # Embedd the cities tensor
    #     embedded_cities = self.initial_embedding(state['cities'])

    #     # Cat the embedded first city, embedded current city and the cities tensor
    #     input_tensor = th.cat((embedded_first_city.unsqueeze(0), embedded_current_city.unsqueeze(0), embedded_cities))

    #     # If n_cities < max_nodes_per_graph, pad the input tensor with zeros and the not visited cities tensor
    #     not_visited_cities = state['not_visited_cities']
    #     if n_cities < self.max_nodes_per_graph:
    #         input_tensor = th.cat((input_tensor, th.zeros(self.max_nodes_per_graph - n_cities, self.embedding_dimension)), dim=0)
    #         not_visited_cities = th.cat((state['not_visited_cities'], th.zeros(self.max_nodes_per_graph - n_cities, dtype=th.bool)), dim=0)

    #     # Reshape the input tensor to be a 1D tensor
    #     input_tensor = input_tensor.view(-1)

    #     # Forward pass
    #     output = self.layers(input_tensor)

    #     # Pass the mask that represents not visited cities to the output tensor. Giving them 0 probaility of being selected
    #     output_masked = output[:-1].masked_fill(~not_visited_cities, -float('inf'))

    #     # Transform the output into probabilities
    #     output_probabilities = th.softmax(output_masked, dim=0)

    #     # Get value
    #     value = output[-1]

    #     return output_probabilities, value