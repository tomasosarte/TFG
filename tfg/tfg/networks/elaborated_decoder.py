import torch as th
import torch.nn as nn

class ElaboratedDecoder(nn.Module):

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
        super(ElaboratedDecoder, self).__init__()

        self.max_nodes_per_graph = max_nodes_per_graph
        self.node_dimension = node_dimension
        self.embedding_dimension = embedding_dimension

        # Input symbol for first step the size of the node dimension
        self.symbol = th.ones(1, node_dimension, dtype=th.float32)*(-1)

        # To map input node features to the embedding space
        self.initial_embedding = nn.Linear(node_dimension, embedding_dimension)
        
        # Layers of the network
        self.layers = nn.Sequential(
            nn.Linear((max_nodes_per_graph + 2) * embedding_dimension + max_nodes_per_graph, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, max_nodes_per_graph + 1),
        )

    def get_state_info(self, state_batch: th.Tensor) -> None:
        """
        Get the information of the state tensor.

        Args:
            state_batch (th.Tensor): A tensor containing the state of the agent in the environment.
        
        Returns:
            num_cities (th.Tensor): The number of cities in the graph.
            batch_size (th.Tensor): The batch size of the state tensor.
            state_shape (th.Tensor): The shape of the state tensor.
            visited_cities (th.Tensor): A tensor containing the visited cities.
            cities (th.Tensor): A tensor containing the cities.
            first_cities (th.Tensor): A tensor containing the first cities.
            current_cities (th.Tensor): A tensor containing the current cities.
            previous_cities (th.Tensor): A tensor containing the previous cities.
        """

        num_cities = state_batch[0][0].type(th.int32)
        batch_size = state_batch.shape[0]
        state_shape = state_batch.shape[1]
        start_cts = 4 + self.max_nodes_per_graph
        visited_cities = state_batch[:, 4:start_cts].type(th.bool)
        cities = state_batch[0][start_cts: start_cts + num_cities*self.node_dimension].view(-1, 2)
        first_cities = state_batch[:, 1].type(th.int32)
        current_cities = state_batch[:, 2].type(th.int32)
        previous_cities = state_batch[:, 3].type(th.int32)

        return num_cities, batch_size, state_shape, visited_cities, cities, first_cities, current_cities, previous_cities

    def get_embeddings(self, cities: th.Tensor, first_cities: th.Tensor, current_cities: th.Tensor) -> th.Tensor:
        """
        Get the embeddings of the cities, first cities and current cities.

        Args:
            cities (th.Tensor): A tensor containing the cities.
            first_cities (th.Tensor): A tensor containing the first cities.
            current_cities (th.Tensor): A tensor containing the current cities.

        Returns:
            embedded_symbol (th.Tensor): The embedding of the symbol.
            embedded_cities (th.Tensor): The embedding of the cities.
            embedded_first_cities (th.Tensor): The embedding of the first cities.
            embedded_current_cities (th.Tensor): The embedding of the current cities.
        """
        # Get embeddings
        embedded_symbol = self.initial_embedding(self.symbol)
        embedded_cities = self.initial_embedding(cities)
        embeddings = th.cat((embedded_cities, embedded_symbol), dim=0)

        # Get the embedded first and current cities
        first_cities_index = first_cities + 1
        current_cities_index = current_cities + 1
        embedded_first_cities = embeddings[first_cities_index]
        embedded_current_cities = embeddings[current_cities_index]

        return embedded_first_cities, embedded_current_cities, embedded_cities
    
    def check_state(self, num_cities: th.Tensor, state_shape, visited_cities: th.Tensor, first_cities: th.Tensor) -> None:
        """
        Check if the state tensor is correct.

        Args:
            num_cities (th.Tensor): The number of cities in the graph.
            state_shape (th.Tensor): The shape of the state tensor.
            visited_cities (th.Tensor): A tensor containing the visited cities.
            first_cities (th.Tensor): A tensor containing the first cities.

        Returns:
            indices_all_visited_cities (th.Tensor): The indices of the states that have all the visited cities.
        """
        assert num_cities <= self.max_nodes_per_graph, "The number of cities in the graph is greater than the maximum number of nodes per graph."
        assert state_shape == (4 + self.max_nodes_per_graph + self.node_dimension*self.max_nodes_per_graph), "The state tensor has the wrong shape."

        # Check that if tensor has all visited cities, first_city_cannot be -1
        all_visited_cities = visited_cities.all(dim=1)
        indices_all_visited_cities = th.nonzero(all_visited_cities).view(-1)
        assert th.all(first_cities[indices_all_visited_cities] != -1), "If all cities are visited, the first city cannot be -1."

        return indices_all_visited_cities

    def change_visited_cities(self, indices_all_visited_cities: th.Tensor, visited_cities: th.Tensor, first_cities: th.Tensor) -> th.Tensor:
        """
        Change the visited cities tensor to not visited the first city.

        Args:
            indices_all_visited_cities (th.Tensor): The indices of the states that have all the visited cities.
            visited_cities (th.Tensor): A tensor containing the visited cities.
            first_cities (th.Tensor): A tensor containing the first cities.

        Returns:
            visited_cities (th.Tensor): The visited cities tensor. 
        """
        if indices_all_visited_cities.shape[0] > 0:
            first_cities_visited = first_cities[indices_all_visited_cities]
            visited_cities[indices_all_visited_cities] = th.ones(visited_cities[indices_all_visited_cities].shape, dtype=th.bool)
            visited_cities[indices_all_visited_cities, first_cities_visited] = 0

        return visited_cities

    def get_padding(self, num_cities: th.Tensor, batch_size: th.Tensor, cities: th.Tensor) -> th.Tensor:
        """
        Get the padding tensor.

        Args:
            num_cities (th.Tensor): The number of cities in the graph.
            batch_size (th.Tensor): The batch size of the state tensor.
            cities (th.Tensor): A tensor containing the cities.

        Returns:
            padded_cities (th.Tensor): The padded cities tensor.
        """
        pad = self.max_nodes_per_graph - num_cities
        padding_tensor = self.symbol.expand(pad, -1).expand(batch_size, -1, -1)
        padded_cities = th.cat((cities.unsqueeze(0).expand(batch_size, -1, -1), padding_tensor), dim=1)
        return padded_cities

    def get_input_tensor(self, embedded_first_cities: th.Tensor, embedded_current_cities: th.Tensor, 
                         embedded_padded_cities: th.Tensor, visited_cities: th.Tensor,
                         batch_size: th.Tensor) -> th.Tensor:
        """
        Get the input tensor of the network.

        Args:
            embedded_first_cities (th.Tensor): The embedding of the first cities.
            embedded_current_cities (th.Tensor): The embedding of the current cities.
            embedded_padded_cities (th.Tensor): The embedding of the padded cities.
            visited_cities (th.Tensor): A tensor containing the visited cities.

        Returns:
            input_tensor (th.Tensor): The input tensor of the network.
        """
        input_tensor = th.cat((embedded_first_cities.unsqueeze(1), embedded_current_cities.unsqueeze(1), embedded_padded_cities), dim=1)
        input_tensor = input_tensor.view(batch_size, -1)
        input_tensor = th.cat((input_tensor, (~visited_cities.view(batch_size, -1)).type(th.float32)), dim=1)

        return input_tensor
    
    def forward(self, state_batch: th.Tensor) -> th.Tensor:
        """
        Forward pass of the network. This function takes a state th.Tensor as input and returns the output of the network.

        Args:
            state (th.Tensor): A tensor containing the state of the agent in the environment.

        Returns:
            th.Tensor: The output of the network.
        """

        # Get info
        num_cities, batch_size, state_shape, visited_cities, cities, first_cities, current_cities, _ = self.get_state_info(state_batch)

        # Check state
        indices_all_visited_cities = self.check_state(num_cities, state_shape, visited_cities, first_cities)

        # Change the visited cities
        visited_cities = self.change_visited_cities(indices_all_visited_cities, visited_cities, first_cities)

        # Get padded cities
        padded_cities = self.get_padding(num_cities, batch_size, cities)

        # Get embeddings
        embedded_first_cities, embedded_current_cities, embedded_padded_cities = self.get_embeddings(padded_cities, first_cities, current_cities)

        # Get input
        input_tensor = self.get_input_tensor(embedded_first_cities, embedded_current_cities, embedded_padded_cities, visited_cities, batch_size)
        
        # Get probabilities and value
        output = self.layers(input_tensor)
        output_masked = output[:, :-1].masked_fill(visited_cities, -float('inf'))
        output_probabilities = th.softmax(output_masked, dim=1)
        value = output[:, -1]

        return output_probabilities, value.view(-1, 1)
