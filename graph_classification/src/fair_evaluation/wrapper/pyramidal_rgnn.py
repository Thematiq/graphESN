import warnings

import numpy as np
import scipy.sparse as sp

from .common import ScikitFriendlyModelWrapper
from torch_geometric.utils import get_laplacian, to_dense_adj, degree

from pyramidal_rgnn.modules.models import RGNN_classifier
from pyramidal_rgnn.modules.pooling import preprocess

def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    # print(A)
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


class PyramidalRGNNWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        self.model = None
        self.pooling = False
        super().__init__(**kwargs)

    def _init_model(self):
        pass

    @staticmethod
    def __normalize_graph(graph):
        e_i, e_a = get_laplacian(graph.edge_index, num_nodes=graph.num_nodes, normalization='sym')
        return to_dense_adj(edge_index=e_i, edge_attr=e_a).squeeze(dim=0).numpy()

    def preprocess(self, data):
        coarse_levels = np.arange(self._params['T'])
        A = [to_dense_adj(g.edge_index, edge_attr=g.edge_attr, max_num_nodes=g.num_nodes).numpy().squeeze() for g in data]
        if data[0].x is None:
            # X = [np.where(g.a.todense() != 0, 1, 0).sum(axis=0)[:, np.newaxis] for g in data]
            X = [degree(g.edge_index[0], g.num_nodes) for g in data]
        else:
            X = [g.x.numpy() for g in data]

        if self._params['mode'] == 'pool':
            A, X, D = preprocess(A, X,
                        coarsening_levels=coarse_levels,
                        pool=self._params['pool'])
            L = [[normalized_adjacency(a_).astype(np.float32) for a_ in a] for a in A]
        else:
            D = None
            L = [normalized_adjacency(a_) for a_ in A]
        return A, X, D, L

    def fit(self, data, y, *args, **kwargs):
        A, X, D, L = self.preprocess(data)
        self.pooling = self._params['mode'] == 'pool'
        self.model = RGNN_classifier(
            embedding_model=self._params['mode'],
            K=self._params['K'],
            T=self._params['T'],
            aggregation=self._params['aggregation'],
            in_scaling=self._params['in_scaling'],
            hid_scaling=self._params['hid_scaling'],
            return_last=self._params['return_last'],
            alpha_ridge=self._params['alpha'],
            readout=self._params['readout'],
            readout_units_factor=self._params['readout_units_factor'],
            n_internal_units=self._params['n_internal_units'],
            spectral_radius=self._params['spectral_radius'],
            max_norm=self._params['max_norm'],
            connectivity=self._params['connectivity'],
            input_weights_mode=self._params['input_weights_mode'],
            noise_level=0.0,
            leak=None,
            circle=False,
        )

        if self.pooling:
            self.model.fit(y, L, X, D)
        else:
            self.model.fit(y, L, X)

        return self

    def predict(self, data, *args, **kwargs):
        A, X, D, L = self.preprocess(data)

        if self.pooling:
            return self.model.transform(L, X, D)
        else:
            return self.model.transform(L, X)
