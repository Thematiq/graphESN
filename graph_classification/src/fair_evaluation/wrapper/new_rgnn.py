from graph_esn.gesn import GroupedDeepGESN
from auto_esn.esn.reservoir.activation import self_normalizing_default, tanh
from auto_esn.esn.reservoir.initialization import WeightInitializer, default_hidden
from graph_esn.readout.aggregator import sum_vertex_features, mean_vertex_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .common import ScikitFriendlyModelWrapper

import torch
import lightgbm as lgb


class GraphESNWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        pass
        # self.__init_model(**self._params)

    def __init_model(self,
                     input_size: int,
                     hidden_size: int,
                     hidden_density: float = 1e-2,
                     activation: str = "tanh",
                     aggregator: str = "mean",
                     num_layers: int = 3,
                     num_groups: int = 1,
                     spectral_radius: float = 5e-2,
                     head: str = "rf",
                     head_n_jobs: int = 1,
                     **activation_kwargs
                     ):
        agg_fn = {
            "mean": mean_vertex_features,
            "sum": sum_vertex_features
        }[aggregator]

        act_fn = {
            "tanh": tanh,
            "sna": self_normalizing_default
        }[activation](**activation_kwargs)

        head_clf = {
            "linear": LogisticRegression(max_iter=10_000),
            "boost": lgb.LGBMClassifier(verbose=-1),
            "rf": RandomForestClassifier(n_estimators=500, n_jobs=head_n_jobs),
        }[head]

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        print(f"Using {self.device}")

        self.model = GroupedDeepGESN(
            input_size=input_size, hidden_size=hidden_size,
            activation=act_fn, aggregator=agg_fn,
            num_layers=num_layers, groups=num_groups,
            initializer=WeightInitializer(
                weight_hh_init=default_hidden(spectral_radius=spectral_radius, density=hidden_density)
            ),
            conv_th=1e-5,
            readout=head_clf,
            device=self.device
        )

    def fit(self, data, y, *args, **kwargs):
        if hasattr(data[0], 'x') and data[0].x is not None:
            input_size = data[0].x.size(dim=1)
        else:
            input_size = 1

        self.__init_model(input_size=input_size, **self._params)
        self.model.fit(self.move_data(data), y)
        return self

    def predict(self, data, *args, **kwargs):
        return self.model(self.move_data(data))
    
    def move_data(self, data):
        return [d.to(self.device) for d in data]

