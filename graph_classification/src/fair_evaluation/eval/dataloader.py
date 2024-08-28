from typing import Tuple, TypeVar
import torch

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from nero.converters.tudataset import tudataset2persisted

Data = TypeVar('Data')
Labels = TypeVar('Labels')


def nero_loader(dataset_name: str, root=None) -> Tuple[Data, Labels]:
    X, y, desc = tudataset2persisted(dataset_name, root_dir=root)
    print(X)
    return X, y, {'desc': desc}


def default_loader(dataset_name: str, root: str = None, append_degree_if_no_features: bool = False, append_unary_if_no_features: bool = False) -> Tuple[Data, Labels]:
    dataset = TUDataset(root=root, name=dataset_name)
    transformers_base = []

    if append_degree_if_no_features and (not hasattr(dataset, 'x') or dataset.x is None):
        raise NotImplementedError()

    if append_unary_if_no_features and (not hasattr(dataset, 'x') or dataset.x is None or dataset.x.size(1) == 0):
        print("Adding constant to node attributes")
        transformers_base.append(T.Constant())
    
    transformer = T.Compose(transformers_base)
    dataset = TUDataset(root=root, name=dataset_name, transform=transformer)

    return dataset, dataset.y, None
