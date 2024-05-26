import torch as th
import torch.nn as nn

class SkipConnection(nn.Module):
    """
    Skip connection module. Allow the gradient to be directly backpropagated to earlier layers by
    skipping some layers. This is useful for deep networks to avoid vanishing gradient problem.

    Benefits:
        1) Alleviateing the vanishing gradient problem: Gradients become increasingly smaller as they
           are backpropagated through the network, and can eventually become so small that they have
           no effect on the weights. Skip connections allow the gradient to be directly backpropagated
           to earlier layers by skipping some layers, which can help to alleviate the vanishing gradient
           problem.

        2) Facilitates the training of deeper networks: By addressing the vanishing gradient problem, 
           skip connections enable the training of much deeper neural networks than would otherwise 
           be feasible, leading to improved performance on a variety of tasks.

        3) Improves gradient flow: Makes easier to the network to learn from input data.

        4) Feature reusability: allows neetowrk to reuse features from earlier layers. Benficial for
           caturing fine grained details.
    """
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self,
                 n_heads: int,
                 embedding_dim: int,
                 feed_forward_hidden: int = 512,
                 normalization: str = 'batch'
                 ) -> None:
        super(MultiHeadAttentionLayer, self).__init__(

        )

class GraphAttentionEncoder(nn.Module):

    def __init__(self, 
                 n_heads: int,
                 embedding_dim: int,
                 n_layers: int,
                 node_dim: int,
                 normalization: str ='batch',
                 feed_forward_hidden: int = 512
                 ) -> None:
        
        super(GraphAttentionEncoder, self).__init__()

        # To map input node features to the embedding space
        self.initial_embedding = nn.Linear(node_dim, embedding_dim) if node_dim is not None else None

        # The multi head attention layers of the attention encoder
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embedding_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))


    def forward(self) -> th.Tensor:
        """
        Forward pass of the BasicNetwork class.

        Returns:
            th.Tensor: The output tensor.
        """
        return th.zeros(1)

