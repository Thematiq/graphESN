import torch.nn.functional as F
import torch

from torch_geometric.data import DataLoader
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool

from .common import ScikitFriendlyModelWrapper


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout_prob):
        super().__init__()

        self.drop_prob = dropout_prob
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GINWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = None
        super().__init__(**kwargs)

    def _init_model(self):
        pass

    def __init_model(self,
                     input_size: int,
                     hidden_size: int,
                     num_classes: int,
                     drop_prob: float,
                     num_layers: int,
                     ):
        self.model = GIN(
            in_channels=input_size,
            hidden_channels=hidden_size,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout_prob=drop_prob,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.025)

    def fit(self, data, y, *args, **kwargs):
        self.__init_model(
            input_size=data[0].x.size(dim=1),
            hidden_size=self._params['hidden_units'],
            num_classes=len(y.unique()),
            drop_prob=self._params['dropout'],
            num_layers=self._params['layers'],
        )
        data_loader = DataLoader(data, batch_size=self._params['batch_size'])


        for epoch in range(self._params['epochs']):
            self.model.train()
            for batch in data_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)

                out = self.model(batch.x, batch.edge_index, batch.batch)
                print(f"Expected size: {batch.batch_size}, batch range {torch.min(batch.batch)} - {torch.max(batch.batch)}, prediction shape {out.shape}, y shape: {batch.y.shape}")
                loss = self.criterion(out, batch.y)
                loss.backward()

                self.optimizer.step()
        
        return self

    @staticmethod
    def __predict_fn(output):
        return output.max(1, keepdim=True)[1].detach().cpu()

    def predict(self, data, *args, **kwargs):
        data_loader = DataLoader(data, batch_size=self._params['batch_size'])
        
        y = []

        for batch in data_loader:
            batch = batch.to(self.device)
            model_out = self.model(batch.x, batch.edge_index, batch.batch)
            pred = self.__predict_fn(model_out)
            y.append(pred)

        return torch.cat(y, 0)
