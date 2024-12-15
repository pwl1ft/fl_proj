from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays
from flwr.proto.transport_pb2 import Scalar

from models import base_model
from models.base_model import Net, train
from utils import set_num_batches_in_state_dict


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 train_loader,
                 val_loader,
                 num_classes) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        state_dict = set_num_batches_in_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.train_loader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.train_loader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, metrics = base_model.test(self.model, self.val_loader, self.device)

        print(f'Client: In evaluate: loss is: {loss}. Metrics are: {metrics}')
        return float(loss), len(self.val_loader), {'metrics': metrics}


def generate_client_fn(train_loaders, val_loaders, num_classes):
    def client_fn(cid: str):
        return FlowerClient(train_loader=train_loaders[int(cid)],
                            val_loader=val_loaders[int(cid)],
                            num_classes=num_classes)

    return client_fn
