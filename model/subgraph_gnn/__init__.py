from torch import nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv
from torch.nn import Identity


class GNN(nn.Module):
    def __init__(self, nin, nout, dropout, bn=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.layer_num = 1
        layers = nn.ModuleList()
        for i in range(self.layer_num):
            layers.append(GCNConv(nin, nin))
            layers.append(nn.BatchNorm1d(nin) if bn else Identity())
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            self.blocks.append(layers)

        self.output_encoder = MLP(in_channels=nin, hidden_channels=nout, out_channels=nout, num_layers=2) if nin != nout else Identity()

    def forward(self, x, edge_index):
        for i in range(self.layer_num):
            for j in range(len(self.blocks[i])):
                if j == 0:
                    x = self.blocks[i][j](x, edge_index)
                else:
                    x = self.blocks[i][j](x)

        x = self.output_encoder(x)
        return x


class SubgraphGCN(nn.Module):
    def __init__(
            self, nin, nout, dropout, hop_dim=16, pooling='mean', embs=(0, 1, 2), mlp_layers=1, param=None
    ):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0

        use_hops = hop_dim > 0
        self.hop_embedder = nn.Embedding(20, hop_dim)

        self.gnn = GNN(nin + hop_dim if use_hops else nin, nin, dropout)

        self.subgraph_transform = MLP(in_channels=nout, hidden_channels=nout, out_channels=nout, num_layers=mlp_layers)
        self.context_transform = MLP(in_channels=nout, hidden_channels=nout, out_channels=nout, num_layers=mlp_layers)

        self.use_hops = use_hops
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())

        self.dropout = dropout
        self.pooling = pooling
        self.embs = embs  # 0-Centroid, 1-Subgraph,  2-Context

        self.param = param

    def forward(self, data, i):
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper[i]]
        combined_subgraphs_edge_index = data.combined_subgraphs[i]
        combined_subgraphs_batch = data.subgraphs_batch[i]

        hop_emb = None
        if self.use_hops:
            hop_emb = self.hop_embedder(data.hop_indicator[i] + 1)
            combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)

        combined_subgraphs_x = self.gnn(combined_subgraphs_x, combined_subgraphs_edge_index)

        centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper[i] == combined_subgraphs_batch)]
        subgraph_x = self.subgraph_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)
        ) if len(self.embs) > 1 else combined_subgraphs_x
        context_x = self.context_transform(
            F.dropout(combined_subgraphs_x, self.dropout, training=self.training)
        ) if len(self.embs) > 1 else combined_subgraphs_x
        if self.use_hops:
            centroid_x = centroid_x * self.gate_mapper_centroid(
                hop_emb[(data.subgraphs_nodes_mapper[i] == combined_subgraphs_batch)])
            subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
            context_x = context_x * self.gate_mapper_context(hop_emb)
        subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
        context_x = scatter(context_x, data.subgraphs_nodes_mapper[i], dim=0, reduce=self.pooling)

        x = [centroid_x, subgraph_x, context_x]
        x = [x[j] for j in self.embs]

        x = sum(x)

        return x
