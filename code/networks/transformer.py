import math
import torch as th
import numpy as np

class CustomSequential(th.nn.Sequential):
    def forward(self, input, mask=None):
        for module in self:
            input = module(input, mask=mask)
        return input
    
class CustomBatchNorm1d(th.nn.BatchNorm1d):
    def forward(self, input, mask=None):
        return super(CustomBatchNorm1d, self).forward(input)
    
class CustomInstanceNorm1d(th.nn.InstanceNorm1d):
    def forward(self, input, mask=None):
        return super(CustomInstanceNorm1d, self).forward(input)     

class CustomLinear(th.nn.Linear):
    def forward(self, input, mask=None):
        return super(CustomLinear, self).forward(input)
    
class CustomReLU(th.nn.ReLU):
    def forward(self, input, mask=None):
        return super(CustomReLU, self).forward(input)

class CustomSkipConnection(th.nn.Module):
    def __init__(self, module):
        super(CustomSkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)

class CustomNormalization(th.nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(CustomNormalization, self).__init__()

        normalizer_class = {
            'batch': CustomBatchNorm1d,
            'instance': CustomInstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):

        if isinstance(self.normalizer, CustomBatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, CustomInstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

# --------------------------------------------------------------

class SkipConnection(th.nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(th.nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = th.nn.Parameter(th.Tensor(n_heads, input_dim, key_dim))
        self.W_key = th.nn.Parameter(th.Tensor(n_heads, input_dim, key_dim))
        self.W_val = th.nn.Parameter(th.Tensor(n_heads, input_dim, val_dim))

        self.W_out = th.nn.Parameter(th.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = th.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = th.matmul(hflat, self.W_key).view(shp)
        V = th.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * th.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = th.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = th.matmul(attn, V)

        out = th.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(th.nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': th.nn.BatchNorm1d,
            'instance': th.nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, th.nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, th.nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(CustomSequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            CustomSkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            CustomNormalization(embed_dim, normalization),
            CustomSkipConnection(
                CustomSequential(
                    CustomLinear(embed_dim, feed_forward_hidden),
                    CustomReLU(),
                    CustomLinear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else CustomLinear(embed_dim, embed_dim)
            ),
            CustomNormalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(th.nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = th.nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = CustomSequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        # assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.reshape(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h, mask=mask)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class Transformer(th.nn.Module):

    def __init__(self, params: dict = {}) -> None:
        """
        Constructor for the BasicNetwork class. This class is used to represent the network that will be used to solve NP-HARD problems.

        Args:
            params (dict): A dictionary containing the parameters for the network.
        
        Returns:
            None
        """
        super(Transformer, self).__init__()

        # Params
        self.max_nodes_per_graph = params.get("max_nodes_per_graph", 20)
        self.node_dim = params.get("node_dimension", 2)
        self.embed_dim = params.get("embed_dim", 4)

        # Symbol
        self.symbol = th.ones(1, self.node_dim, dtype=th.float32)*(-1)

        self.graph_encoder = GraphAttentionEncoder(
            n_heads=params.get("n_heads", 4),
            embed_dim=self.embed_dim,
            n_layers=params.get("n_layers", 3),
            node_dim=self.node_dim,
            normalization=params.get("normalization", 'batch'),
            feed_forward_hidden=params.get("feed_forward_hidden", 512)
        )

        self.input_size = 2 + (self.max_nodes_per_graph + 1)*self.embed_dim + self.max_nodes_per_graph
        self.output_size = self.max_nodes_per_graph + 1
        self.decoder = th.nn.Sequential(
            th.nn.Linear(self.input_size, 128), th.nn.ReLU(),
            th.nn.Linear(128, 512), th.nn.ReLU(),
            th.nn.Linear(512, 128), th.nn.ReLU(),
            th.nn.Linear(128, self.output_size),
        )


    def forward(self, states: th.Tensor) -> th.Tensor:

        num_instances = states.shape[0]

        # Get cities
        init_cities = 4 + self.max_nodes_per_graph
        end_cities = init_cities + self.max_nodes_per_graph*self.node_dim
        cities = states[:, init_cities:end_cities].reshape(num_instances, self.max_nodes_per_graph, self.node_dim)

        # Get visited mask
        visited_mask = states[:, 4:4+self.max_nodes_per_graph]
        visited_mask = (visited_mask == 1)
        visited_mask = visited_mask.unsqueeze(1) | visited_mask.unsqueeze(2)

        # Get graph encoding
        embedded_cities, mean = self.graph_encoder(cities, mask=visited_mask)
        embedded_cities = embedded_cities.reshape(num_instances, -1)
        graph_encoding = th.cat((mean, embedded_cities), dim=1)

        # Get first& previous cities index
        first_cities_idx = states[:, 1]
        current_cities_idx = states[:, 2]

        # Get visited mask
        visited_mask = states[:, 4:4+self.max_nodes_per_graph]

        # Get decoder input
        decoder_input = th.cat((first_cities_idx.unsqueeze(1), current_cities_idx.unsqueeze(1), graph_encoding, visited_mask), dim=1)

        return self.decoder(decoder_input)
       
    
if __name__ == "__main__":

    model = Transformer()
    padding = th.ones(10, 2)*(-1)

    # Get cities
    cities10_0 = th.load(f"../training/tsp/size_{10}/instance_{0}.pt") 
    cities10_0 = th.cat((cities10_0, padding), dim=0)
    cities10_1 = th.load(f"../training/tsp/size_{10}/instance_{1}.pt")
    cities10_1 = th.cat((cities10_1, padding), dim=0)
    cities20_0 = th.load(f"../training/tsp/size_{20}/instance_{0}.pt")
    cities20_1 = th.load(f"../training/tsp/size_{20}/instance_{1}.pt")
    all_cities = th.cat((cities10_0, cities10_1, cities20_0, cities20_1), dim=0).reshape(4, -1)

    # visited_cities tensor
    visited_cities_10 = th.cat((th.zeros(1, 10), th.ones(1, 10)), dim=1)
    visited_cities_20 = th.zeros(1, 20)
    visited_cities = th.cat((visited_cities_10, visited_cities_10, visited_cities_20, visited_cities_20), dim=0)

    # Metadata
    metadata = th.tensor([[10, -1, -1, -1], [10, -1, -1, -1], [20, -1, -1, -1], [20, -1, -1, -1]])

    # Get batch of states
    states = th.cat((metadata, visited_cities, all_cities), dim=1)
    
    # Forward pass
    output = model(states)

    # Test MultiheadAttentionLayer with mask
    all_cities = all_cities.reshape(4, -1, 2)

    # Cast all cities as boolean
    mask = (visited_cities == 1)
    mask[0][2] = True
    # Transform visited cities mask to negative adjacency matrix
    mask_expanded = mask.unsqueeze(1) | mask.unsqueeze(2)

    attention = MultiHeadAttention(n_heads=4, input_dim=2, embed_dim=4)
    attention(q=all_cities, mask=mask_expanded)
    
