import torch

from torch import Tensor

from typing import Callable


LatentFeatureAggregator = Callable[[Tensor], Tensor]


def mean_vertex_features(X: Tensor) -> Tensor:
    return torch.mean(X, dim=0)


def sum_vertex_features(X: Tensor) -> Tensor:
    return torch.sum(X, dim=0)
