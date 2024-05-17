import torch as th
import torch.nn as nn

class MoreBasicNetwork(nn.Module):

    def __init__(self, params: dict = {}) -> None:
        """
        Constructor for the BasicNetwork class. This class is used to represent the network that will be used to solve NP-HARD problems.

        Args:
            max_nodes_per_graph (int): The maximum number of nodes that a graph can have.
            node_dimension (int): The dimension of the node features.
            embedding_dimension (int): The dimension of the node embeddings.
        
        Returns:
            None
        """
        super(MoreBasicNetwork, self).__init__()

        self.max_nodes_per_graph = params.get("max_nodes_per_graph", 10)
        self.node_dimension = params.get("node_dimension", 2)
        
        self.input_size = 4 + self.max_nodes_per_graph + self.max_nodes_per_graph*self.node_dimension
        self.output_size = self.max_nodes_per_graph + 1
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, self.output_size))
    
    def forward(self, state_batch: th.Tensor) -> th.Tensor:
        assert state_batch.shape[1] == self.input_size, "Input size does not match the expected size."
        return self.layers(state_batch)