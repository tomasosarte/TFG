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
        self.symbol = th.ones(1, node_dimension, dtype=th.float32)*(-1)

        # To map input node features to the embedding space
        self.initial_embedding = nn.Linear(node_dimension, embedding_dimension)
        
        # Layers of the network
        self.layers = nn.Sequential(
            nn.Linear((max_nodes_per_graph + 2) * embedding_dimension, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, max_nodes_per_graph + 1)
        )

    def forward(self, state_batch: th.Tensor) -> th.Tensor:
        """
        Forward pass of the network. This function takes a state th.Tensor as input and returns the output of the network.

        Args:
            state (th.Tensor): A tensor containing the state of the agent in the environment.

        Returns:
            th.Tensor: The output of the network.
        """
        num_cities = state_batch[0][0].type(th.int32)
        batch_size = state_batch.shape[0]
        state_shape = state_batch.shape[1]
        assert num_cities <= self.max_nodes_per_graph, "The number of cities in the graph is greater than the maximum number of nodes per graph."
        assert state_shape == (4 + self.max_nodes_per_graph + self.node_dimension*self.max_nodes_per_graph), "The state tensor has the wrong shape."

        start_cts = 4 + self.max_nodes_per_graph
        visited_cities = state_batch[:, 4:start_cts].type(th.bool)
        cities = state_batch[0][start_cts: start_cts + num_cities*self.node_dimension].view(-1, 2)
        first_cities = state_batch[:, 1].type(th.int32)
        current_cities = state_batch[:, 2].type(th.int32)
        previous_cities = state_batch[:, 3].type(th.int32)

        # Check that if tensor has all visited cities, first_city_cannot be -1
        all_visited_cities = visited_cities.all(dim=1)
        indices_all_visited_cities = th.nonzero(all_visited_cities).view(-1)
        assert th.all(first_cities[indices_all_visited_cities] != -1), "If all cities are visited, the first city cannot be -1."

        # Change the visited cities tensor to not visited the first city
        if indices_all_visited_cities.shape[0] > 0:
            first_cities_visited = first_cities[indices_all_visited_cities]
            visited_cities[indices_all_visited_cities] = th.ones(visited_cities[indices_all_visited_cities].shape, dtype=th.bool)
            visited_cities[indices_all_visited_cities, first_cities_visited] = 0

        # Get embeddings
        embedded_symbol = self.initial_embedding(self.symbol)
        embedded_cities = self.initial_embedding(cities)
        embeddings = th.cat((embedded_cities, embedded_symbol), dim=0)

        # Get first city and current city index in the embeddings
        first_cities_index = first_cities + 1
        current_cities_index = current_cities + 1

        # Get the embedded first and current cities
        embedded_first_cities = embeddings[first_cities_index]
        embedded_current_cities = embeddings[current_cities_index]

        # If n_cities < max_nodes_per_graph, pad the input tensor with zeros and the not visited cities tensor
        pad = self.max_nodes_per_graph - num_cities
        
        # Get pad embedded symbols and concat them with cities
        padding_tensor = embedded_symbol.expand(pad, -1).expand(batch_size, -1, -1)
        padded_embedded_cities = th.cat((embedded_cities.unsqueeze(0).expand(batch_size, -1, -1), padding_tensor), dim=1)

        # Get input tensor
        input_tensor = th.cat((embedded_first_cities.unsqueeze(1), embedded_current_cities.unsqueeze(1), padded_embedded_cities), dim=1)
        input_tensor = input_tensor.view(batch_size, -1)
        
        # Forward pass
        output = self.layers(input_tensor)

        # Pass the mask that represents visited cities to the output tensor. Giving them 0 probaility of being selected
        output_masked = output[:, :-1].masked_fill(visited_cities, -float('inf'))

        # Transform the output into probabilities
        output_probabilities = th.softmax(output_masked, dim=1)

        # Get value
        value = output[:, -1]

        return output_probabilities, value
