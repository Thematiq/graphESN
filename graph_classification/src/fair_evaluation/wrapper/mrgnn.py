import torch

from torch_geometric.loader import DataLoader
from MRGNN.model.MRGNN import MRGNN
from tqdm import trange

from .common import ScikitFriendlyModelWrapper


class MRGNNWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = None
        # print(f"MRGNN device: {self.device}")
        super().__init__(**kwargs)

    def _init_model(self):
        pass

    def __init_model(self,
                     input_size: int,
                     hidden_size: int,
                     n_classes: int,
                     drop_prob: float,
                     max_k: int,
                     output: str,
                     learning_rate: float,
                     weight_decay: float):
        self.model = MRGNN(
            in_channels=input_size,
            out_channels=hidden_size,
            n_class=n_classes,
            drop_prob=drop_prob,
            max_k=max_k,
            output=output,
        ).to(self.device)

        train_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        self.optimizer = torch.optim.AdamW(train_params, lr=learning_rate, weight_decay=weight_decay)

    def fit(self, data, y, *args, **kwargs):
        input_size = data[0].x.size(dim=1)
        n_classes = len(y.unique())

        self.__init_model(
            input_size=input_size,
            hidden_size=self._params['n_units'],
            n_classes=n_classes,
            drop_prob=self._params['drop_prob'],
            max_k=self._params['max_k'],
            output=self._params['output'],
            learning_rate=self._params['lr'],
            weight_decay=self._params['weight_decay']
        )

        if 'fit_params' in kwargs and 'copy_data' in kwargs['fit_params']:
            data = [x.clone() for x in data]

        if self._params['adjacency_matrix'] == 'L':
            data = [self.model.get_TANH_resevoir_L(x.to(self.device)).to('cpu') for x in data]
        else:
            data = [self.model.get_TANH_resevoir_A(x.to(self.device)).to('cpu') for x in data]

        data_loader = DataLoader(data, batch_size=self._params['batch_size'])


        for epoch in range(self._params['n_epochs']):
            self.model.train()
            for batch in data_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                out = self.model.readout_fw(batch)

                loss = self.criterion(out, batch.y)

                loss.backward()
                self.optimizer.step()

        return self

    @staticmethod
    def __predict_fn(output):
        return output.max(1, keepdim=True)[1].detach().cpu()

    def predict(self, data, *args, **kwargs):
        if self._params['adjacency_matrix'] == 'L':
            data = [self.model.get_TANH_resevoir_L(x.to(self.device)) for x in data]
        else:
            data = [self.model.get_TANH_resevoir_A(x.to(self.device)) for x in data]


        data_loader = DataLoader(data, batch_size=self._params['batch_size'])

        y = []

        for batch in data_loader:
            batch = batch.to(self.device)
            model_out = self.model.readout_fw(batch)
            pred = self.__predict_fn(model_out)
            y.append(pred)

        return torch.cat(y, 0)
