import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


def add_random_walk_se(data, walk_length=20, attr_name='se'):
    assert data.edge_index is not None
    N = data.num_nodes
    assert N is not None

    structural_encodings = []
    for i in range(7):
        row, col = data.seven_edge_index[i]
        if data.edge_weight is None:
            value = torch.ones(data.seven_edge_index[i].shape[1], device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.NO_MKL:  # pragma: no cover
            adj = to_torch_coo_tensor(data.seven_edge_index[i], value, size=data.size())
        else:
            adj = to_torch_csr_tensor(data.seven_edge_index[i], value, size=data.size())

        def get_se(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        se_list = [get_se(out)]
        for _ in range(walk_length - 1):
            out = out @ adj
            se_list.append(get_se(out))

        se = torch.stack(se_list, dim=-1)

        structural_encodings.append(se)

    data.__setattr__(attr_name, structural_encodings)


def add_laplacian_eigenvector_pe(data, k, attr_name='pe', is_undirected=False):
    assert data.edge_index is not None
    num_nodes = data.num_nodes
    assert num_nodes is not None

    SPARSE_THRESHOLD = 100

    positional_encodings = []
    for i in range(7):
        edge_index, edge_weight = get_laplacian(
            data.seven_edge_index[i],
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=k + 1,
                which='SR' if not is_undirected else 'SA',
                return_eigenvectors=True,
            )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (k,))
        pe *= sign

        positional_encodings.append(pe)

    data.__setattr__(attr_name, positional_encodings)
