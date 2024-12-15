import torch
import torch.cuda
from omegaconf import DictConfig

from models.base_model import Net, test
from utils import Utils


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            'lr': config.lr,
            'momentum': config.momentum,
            'local_epochs': config.local_epochs,
            # 'server_round': server_round
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, test_loader):
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Utils.set_parameters(model, parameters)

        loss, metrics = test(model, test_loader, device)

        return loss, {
            'metrics': metrics,
            # 'server round': server_round
        }

    return evaluate_fn
