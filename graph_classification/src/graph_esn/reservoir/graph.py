import torch
from torch import nn, Tensor
from torch.nn.functional import linear
from typing import Union, List

from auto_esn.esn.reservoir.activation import Activation, tanh, self_normalizing_default
from auto_esn.esn.reservoir.cell import ESNCellBase
from auto_esn.esn.reservoir.initialization import WeightInitializer


class GraphESNCell(ESNCellBase):
    def __init__(self, node_attr_size: int,
                 hidden_size: int,
                 initializer: WeightInitializer,
                 activation: Activation = tanh(),
                 conv_th: float = 1e-5,
                 max_iter: int = 50):
        super().__init__(
            input_size=node_attr_size,
            hidden_size=hidden_size,
            bias=False,
            initializer=initializer,
            requires_grad=False,
            init=True
        )
        self.hx = None
        self.wiu = None
        self.gpu_enabled = False
        self.activation = activation
        self.conv_th = conv_th
        self.max_iter = max_iter

    def reset_hidden(self):
        self.hx = None
        self.wiu = None

    def to_cuda(self):
        self.to('cuda:0')
        self.gpu_enabled = True


    def move_to(self, device):
        self.to(device)

    def _step_forward(self, L: Tensor) -> Tensor:
        z = self.wiu + torch.mm(L, linear(self.hx, self.weight_hh))
        state = self.activation(z, self.hx)
        self.hx = state
        return state

    def forward(self, X: Tensor, L: Tensor) -> Tensor:
        if self.hx is None:
            self.hx = torch.zeros((X.size(dim=0), self.hidden_size),
                                  device=L.device)
            self.wiu = linear(X, self.weight_ih)

        last_res = None
        res = None

        for _ in range(self.max_iter):
            res = self._step_forward(L)

            if last_res is not None and \
                    torch.norm(last_res - res) < self.conv_th:
                break

            last_res = res

        return res


class DeepGESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 initializer: WeightInitializer = WeightInitializer(),
                 num_layers: int = 1,
                 activation: Activation = 'default',
                 conv_th: float = 1e-5,
                 max_iter: int = 50):
        super().__init__()
        if type(activation) != list:
            activation = [activation] * num_layers
        else:
            activation = activation

        self.output_size = hidden_size
        self.layers = [GraphESNCell(input_size, hidden_size,
                                    initializer, activation[0], conv_th, max_iter)]
        if num_layers > 1:
            self.layers += [GraphESNCell(hidden_size, hidden_size,
                                         initializer, activation[i], conv_th, max_iter)
                            for i in range(1, num_layers)]
        self.gpu_enabled = False

    def __forward_pyramidal(self, X: Tensor, L: List[Tensor]) -> Tensor:
        res = []
        cell_input = X
        for i, (gesn_cell, layer_L) in enumerate(zip(self.layers, L)):
            cell_input = gesn_cell(cell_input, layer_L)
            res.append(cell_input)
        return torch.cat(res, dim=-1)

    def __forward_normal(self, X: Tensor, L: Tensor) -> Tensor:
        res = []
        cell_input = X
        for i, gesn_cell in enumerate(self.layers):
            cell_input = gesn_cell(cell_input, L)
            res.append(cell_input)
        return torch.cat(res, dim=-1)

    def forward(self, X: Tensor, L: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(L, list):
            return self.__forward_pyramidal(X, L)
        return self.__forward_normal(X, L)

    def get_hidden_size(self):
        return sum([l.hidden_size for l in self.layers])

    def reset_hidden(self):
        for layer in self.layers:
            layer.reset_hidden()

    def to_cuda(self):
        for layer in self.layers:
            layer.to_cuda()
        self.gpu_enabled = True

    def move_to(self, device):
        for layer in self.layers:
            layer.move_to(device)


class GroupOfGESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 groups, activation=self_normalizing_default(),
                 initializer: WeightInitializer = WeightInitializer(),
                 conv_th: float = 1e-5, max_iter: int = 50):
        super(GroupOfGESNCell, self).__init__()
        num_groups = groups if type(groups) == int else len(groups)
        self.output_size = hidden_size
        if type(activation) != list:
            activation = [activation] * num_groups
        else:
            activation = activation
        if type(groups) != int:
            self.groups = groups
        else:
            self.groups = [GraphESNCell(input_size, hidden_size,
                                        initializer, activation[i], conv_th, max_iter)
                           for i in range(groups)]

        self.hidden_size = hidden_size
        self.gpu_enabled = False

    def __forward_pyramid(self, X: Tensor, L: List[Tensor]) -> Tensor:
        assert isinstance(self.groups[0], DeepGESNCell), "Inner model cells must support pyramidal input"
        assert len(L) == len(self.groups)
        res = []
        for i, (gesn_cell, layer_L) in enumerate(zip(self.groups, L)):
            res.append(gesn_cell(X, layer_L))
        return torch.cat(res, dim=-1)

    def __forward_normal(self, X: Tensor, L: Tensor) -> Tensor:
        res = []
        for i, gesn_cell in enumerate(self.groups):
            res.append(gesn_cell(X, L))
        return torch.cat(res, dim=-1)

    def forward(self, X: Tensor, L: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(L, list):
            return self.__forward_pyramid(X, L)
        return self.__forward_normal(X, L)

    def get_hidden_size(self):
        return sum([cell.get_hidden_size() for cell in self.groups])

    def reset_hidden(self):
        for group in self.groups:
            group.reset_hidden()

    def to_cuda(self):
        for group in self.groups:
            group.to_cuda()
        self.gpu_enabled = True

    def move_to(self, device):
        for group in self.groups:
            group.move_to(device)


class DeepGESNPoolCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 pooling,
                 initializer: WeightInitializer = WeightInitializer(),
                 num_layers: int = 1,
                 activation: Activation = 'default',
                 conv_th: float = 1e-5,
                 max_iter: int = 50):
        super().__init__()
        if type(activation) != list:
            activation = [activation] * num_layers
        else:
            activation = activation
        if type(pooling) != list:
            self.pooling = [pooling] * num_layers
        else:
            self.pooling = pooling

        self.output_size = hidden_size
        self.layers = [GraphESNCell(input_size, hidden_size,
                                    initializer, activation[0], conv_th, max_iter)]
        if num_layers > 1:
            self.layers += [GraphESNCell(hidden_size, hidden_size,
                                         initializer, activation[i], conv_th, max_iter)
                            for i in range(1, num_layers)]
        self.gpu_enabled = False

    def __forward_pyramidal(self, X: Tensor, L: List[Tensor]) -> Tensor:
        raise NotImplementedError()

    def __forward_normal(self, X: Tensor, L: Tensor) -> Tensor:
        res = []
        cell_input = X
        for i, gesn_cell in enumerate(self.layers):
            cell_input = gesn_cell(cell_input, L)
            res.append(cell_input)
        return torch.cat(res, dim=-1)

    def forward(self, X: Tensor, L: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(L, list):
            return self.__forward_pyramidal(X, L)
        return self.__forward_normal(X, L)

    def get_hidden_size(self):
        return sum([l.hidden_size for l in self.layers])

    def reset_hidden(self):
        for layer in self.layers:
            layer.reset_hidden()

    def to_cuda(self):
        for layer in self.layers:
            layer.to_cuda()
        self.gpu_enabled = True

    def move_to(self, device):
        for layer in self.layers:
            layer.move_to(device)