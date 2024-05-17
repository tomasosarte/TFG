import torch as th
import torch.nn as nn

class Embedder(th.nn.Module):
    def __init__(self, params: dict = {}):
        super(Embedder, self).__init__()

        # Init vars
        self.input_dim = params.get('node_dimension', 2)
        self.embed_dim = params.get('embed_dim', 4)
        self.max_nodes_per_graph = params.get('max_nodes_per_graph', 20)
        self.symbol = th.ones(1, self.input_dim, dtype=th.float32)*(-1)

        # Init layers
        self.embedding_layer = th.nn.Linear(self.input_dim, self.embed_dim)
        
    def forward(self, states: th.Tensor) -> th.Tensor:
        
        num_transitions = states.shape[0]

        # Get cities
        init_cities = 4+self.max_nodes_per_graph
        cities = states[:, init_cities:init_cities+self.max_nodes_per_graph*self.input_dim].reshape(-1, self.input_dim)
        cities = th.cat((self.symbol, cities), dim=0)

        # Embed cities
        embedded_cities = self.embedding_layer(cities)

        # Get embeddings of first city, current city and previous city
        first_cities_idx = (states[:, 1] + 1).long()
        current_cities_idx = (states[:, 2] + 1).long()
        previous_cities_idx = (states[:, 3] + 1).long()

        first_cities = embedded_cities[first_cities_idx]
        current_cities = embedded_cities[current_cities_idx]
        previous_cities = embedded_cities[previous_cities_idx]

        # Get cities embedded
        cities_embedded = embedded_cities[1:].view(num_transitions, -1)

        # Concatenate embeddings
        embeddings = th.cat((first_cities, current_cities, previous_cities, cities_embedded), dim=1)

        return embeddings

class EmbedderNetwork(nn.Module):

    def __init__(self, params: dict = {}) -> None:
        """
        Constructor for the BasicNetwork class. This class is used to represent the network that will be used to solve NP-HARD problems.

        Args:
            params (dict): A dictionary containing the parameters for the network.
        """
        super(EmbedderNetwork, self).__init__()

        # Params
        self.max_nodes_per_graph = params.get("max_nodes_per_graph", 10)
        self.embedding_dimension = params.get("embed_dim", 4)

        # Embedder
        self.embedder = Embedder(params)
        
        # Decoder
        self.input_size = (3 + self.max_nodes_per_graph) * self.embedding_dimension + self.max_nodes_per_graph
        self.output_size = self.max_nodes_per_graph + 1
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, self.output_size)
        )
    
    def forward(self, state_batch: th.Tensor) -> th.Tensor:

        embeddings = self.embedder(state_batch)
        visited_cities = state_batch[:, 4:4+self.max_nodes_per_graph]
        input_batch = th.cat((embeddings, visited_cities), dim=1)

        return self.layers(input_batch)
    
if __name__ == "__main__":

    model = EmbedderNetwork(params={"max_nodes_per_graph": 20, "embed_dim": 4})
    padding = th.ones(10, 2)*(-1)

    # Get cities
    cities10_0 = th.load(f"../training/tsp/size_{10}/instance_{0}.pt") 
    cities10_0 = th.cat((cities10_0, padding), dim=0)
    cities10_1 = th.load(f"../training/tsp/size_{10}/instance_{1}.pt")
    cities10_1 = th.cat((cities10_1, padding), dim=0)
    cities20_0 = th.load(f"../training/tsp/size_{20}/instance_{0}.pt")
    cities20_1 = th.load(f"../training/tsp/size_{20}/instance_{1}.pt")
    all_cities = th.cat((cities10_0, cities10_1, cities20_0, cities20_1), dim=0).view(4, -1)

    # visited_cities tensor
    visited_cities_10 = th.cat((th.zeros(1, 10), th.ones(1, 10)), dim=1)
    visited_cities_20 = th.zeros(1, 20)
    visited_cities = th.cat((visited_cities_10, visited_cities_10, visited_cities_20, visited_cities_20), dim=0)

    # Metadata
    metadata = th.tensor([[10, -1, -1, -1], [10, -1, -1, -1], [20, -1, -1, -1], [20, -1, -1, -1]])

    # Get batch of states
    states = th.cat((metadata, visited_cities, all_cities), dim=1)
    
    # Forward pass
    outuput = model(states)