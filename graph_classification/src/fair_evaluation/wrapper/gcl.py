from .common import ScikitFriendlyModelWrapper

from GCL.models import DualBranchContrast
import torch.nn.functional as F
import torch
from GCL import losses as L
from GCL import augmentors as A

from torch import nn

from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2
    

class GCLWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        self.gconv = None
        self.encoder = None
        self.contrast = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None
        super().__init__(**kwargs)

    def _init_model(self):
        pass

    def __init_model(self, num_features, hidden_size, num_layers):
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                            A.NodeDropping(pn=0.1),
                            A.FeatureMasking(pf=0.1),
                            A.EdgeRemoving(pe=0.1)], 1)

        self.gconv = GConv(input_dim=num_features, hidden_dim=hidden_size, num_layers=num_layers).to(self.device)
        self.encoder = Encoder(encoder=self.gconv, augmentor=(aug1, aug2)).to(self.device)
        self.contrast = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(self.device)
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=0.01)
        self.head = RandomForestClassifier(n_estimators=1000)
        

    def fit(self, data, y, *args, **kwargs):
        self.__init_model(
            num_features=data[0].x.size(dim=1),
            hidden_size=self._params['hidden_size'],
            num_layers=self._params['layers'],
        )

        data_loader = DataLoader(data, batch_size=128)
        for epoch in range(self._params['epochs']):
            self.encoder.train()
            for batch in data_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                _, _, _, _, g1, g2 = self.encoder(batch.x, batch.edge_index, batch.batch)
                g1, g2 = [self.encoder.encoder.project(g) for g in [g1, g2]]
                loss = self.contrast(g1=g1, g2=g2, batch=batch.batch)
                loss.backward()
                self.optimizer.step()

        self.encoder.eval()
        # Train head
        x = []

        for batch in data_loader:
            batch = batch.to(self.device)

            _, g, _, _, _, _ = self.encoder(batch.x, batch.edge_index, batch.batch)

            x.append(g)

        x = torch.cat(x, 0).detach().cpu()
        self.head.fit(x, y)

        return self
    
    def predict(self, data, *args, **kwargs):
        self.encoder.eval()
        data_loader = DataLoader(data, batch_size=128)

        x = []

        for batch in data_loader:
            batch = batch.to(self.device)

            _, g, _, _, _, _ = self.encoder(batch.x, batch.edge_index, batch.batch)

            x.append(g)

        x = torch.cat(x, 0).detach().cpu()

        return self.head.predict(x)
    
