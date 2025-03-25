import re

import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')] + 'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index
    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]  # each one contains <= i hop masks
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix

    return node_mask, hop_indicator


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]

    return combined_subgraphs


from torch_cluster import random_walk


def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1, repeat=1, cal_hops=True, max_hops=10):
    """
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures
            that we are less likely to sample an already visited node in the following two steps.
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
            if q > 1, the random walk is biased towards nodes close to node t.
            if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
        p, q ∈ {0.25, 0.50, 1, 2, 4}.
        Typical values:
        Fix p and tune q

        repeat: restart the random walk many times and combine together for the result

    """
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [random_walk(row, col,
                         start=start,
                         walk_length=walk_length,
                         p=p, q=q,
                         num_nodes=num_nodes) for _ in range(repeat)]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    # print(walk.shape)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length + 1) * repeat), walk.reshape(-1)] = True
    if cal_hops:  # this is fast enough
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
            if hop_indicator[node_mask].min() != -1:
                break
        return node_mask, hop_indicator
    return node_mask, None


def hops_to_selected_nodes(edge_index, selected_nodes, num_nodes=None):
    row, col = edge_index
    if num_nodes is None:
        num_nodes = 1 + edge_index.max()
    hop_indicator = row.new_full((num_nodes,), -1)
    bipartitie_indicator = row.new_full(row.shape, -1)
    hop_indicator[selected_nodes] = 0
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    selected_nodes = (hop_indicator == 0)
    i = 1
    while hop_indicator.min() < 0:
        source_near_edges = selected_nodes[row]
        node_mask.fill_(False)
        node_mask[col[source_near_edges]] = True
        selected_nodes = (hop_indicator == -1) & node_mask
        bipartitie_between_source_target = source_near_edges & selected_nodes[col]
        bipartitie_indicator[bipartitie_between_source_target] = i
        hop_indicator[selected_nodes] = i
        i += 1

    return hop_indicator, bipartitie_indicator


def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges, hop_indicator


def extract_subgraphs(edge_index, num_nodes, num_hops, walk_length=0, p=1, q=1, repeat=1, sparse=False):
    if walk_length > 0:
        node_mask, hop_indicator = random_walk_subgraph(
            edge_index, num_nodes, walk_length, p=p, q=q, repeat=repeat, cal_hops=True
        )
    else:
        node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes, num_hops)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]]  # N x E dense mask matrix
    if not sparse:
        return node_mask, edge_mask, hop_indicator
    else:
        return to_sparse(node_mask, edge_mask, hop_indicator)


def transform_subgraphs(graph):
    # extract subgraphs
    graph = SubgraphsData(**{k: v for k, v in graph})

    graph.subgraphs_batch = []
    graph.subgraphs_nodes_mapper = []
    graph.subgraphs_edges_mapper = []
    graph.combined_subgraphs = []
    graph.hop_indicator = []
    graph.__num_nodes__ = graph.num_nodes

    for i in range(7):
        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(
            graph.seven_edge_index[i], graph.num_nodes, num_hops=1, walk_length=0, p=1, q=1, repeat=5
        )
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
            subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense
        )

        combined_subgraphs = combine_subgraphs(
            graph.seven_edge_index[i], subgraphs_nodes, subgraphs_edges, num_selected=graph.num_nodes, num_nodes=graph.num_nodes
        )

        graph.subgraphs_batch.append(subgraphs_nodes[0])
        graph.subgraphs_nodes_mapper.append(subgraphs_nodes[1])
        graph.subgraphs_edges_mapper.append(subgraphs_edges[1])
        graph.combined_subgraphs.append(combined_subgraphs)
        graph.hop_indicator.append(hop_indicator)

    return graph
