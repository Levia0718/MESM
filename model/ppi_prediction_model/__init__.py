import torch
from torch import nn
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GPSConv, GCNConv

from model.subgraph_gnn import SubgraphGCN


class MESM(nn.Module):
    def __init__(self, hidden=1024, class_num=7, se_dim=20, param=None):
        super(MESM, self).__init__()
        self.param = param

        # SE
        self.se_blocks = nn.ModuleList()
        for i in range(class_num):
            layers = nn.ModuleList()
            layers.append(GPSConv(hidden + se_dim, GCNConv(hidden + se_dim, hidden + se_dim), heads=4, attn_type='multihead', attn_kwargs={'dropout': 0.5}))
            layers.append(nn.Linear(hidden + se_dim, hidden))
            self.se_blocks.append(layers)

        # GAT
        self.Graph_Neural_Network_blocks = nn.ModuleList()
        self.layer_num = 1
        for i in range(class_num):
            layers = nn.ModuleList()
            for j in range(self.layer_num):
                layers.append(GAT(hidden, hidden, 1, act=nn.ReLU(), act_first=True, norm=nn.BatchNorm1d(hidden)))

            self.Graph_Neural_Network_blocks.append(layers)

        # SubgraphGCN
        self.subgraph_blocks = nn.ModuleList()
        for i in range(class_num):
            sub_layers = nn.ModuleList()
            sub_layers.append(SubgraphGCN(hidden, hidden, 0.2, 0, 'mean', (0, 1, 2), 2, param=param))

            global_layers = nn.ModuleList()
            global_layers.append(GCNConv(hidden, hidden))

            layers = nn.ModuleList()
            layers.append(nn.BatchNorm1d(hidden))
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5))

            tmp_layer = nn.ModuleList()
            tmp_layer.append(sub_layers)
            tmp_layer.append(global_layers)
            tmp_layer.append(layers)
            self.subgraph_blocks.append(tmp_layer)

        # Classifier
        self.classifier = nn.ModuleList()
        self.classifier_layer_num = 6
        for i in range(self.classifier_layer_num):
            if i != 5:
                self.classifier.append(
                    nn.Linear(
                        int(hidden * ((class_num + 1) / (2 ** i))), int(hidden * ((class_num + 1) / (2 ** (i+1))))
                    )
                )
            else:
                self.classifier.append(
                    nn.Linear(
                        int(hidden * ((class_num + 1) / (2 ** i))), class_num
                    )
                )

    def graph_encoder(self, graph):
        x = graph.x.clone()

        output = [x]
        for i in range(len(self.Graph_Neural_Network_blocks)):
            tmp = x

            tmp = torch.cat((tmp, graph.se[i]), 1)
            for j in range(len(self.se_blocks[i])):
                if j == 0:
                    tmp = self.se_blocks[i][j](tmp, graph.seven_edge_index[i])
                else:
                    tmp = self.se_blocks[i][j](tmp)

            for j in range(self.layer_num):
                tmp = self.Graph_Neural_Network_blocks[i][j](tmp, graph.seven_edge_index[i])

            graph.x = tmp
            tmp_sub = None
            for j in range(len(self.subgraph_blocks[i][0])):
                if j == 0:
                    tmp_sub = self.subgraph_blocks[i][0][j](graph, i)
                else:
                    tmp_sub = self.subgraph_blocks[i][0][j](tmp_sub)
            for j in range(len(self.subgraph_blocks[i][1])):
                if j == 0:
                    tmp = self.subgraph_blocks[i][1][j](tmp, graph.seven_edge_index[i])
                else:
                    tmp = self.subgraph_blocks[i][1][j](tmp)
            tmp = tmp + tmp_sub
            for j in range(len(self.subgraph_blocks[i][2])):
                tmp = self.subgraph_blocks[i][2][j](tmp)

            output.append(tmp)

        x = torch.cat(output, dim=1)
        return x

    def ppi_generator(self, x, node_id):
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x = torch.mul(x1, x2)

        for i in range(self.classifier_layer_num):
            x = self.classifier[i](x)

        return x

    def forward(self, graph, train_edge_id):
        graph = graph.clone()
        node_id = graph.edge_index[:, train_edge_id]

        x = self.graph_encoder(graph)

        x = self.ppi_generator(x, node_id)

        return x
