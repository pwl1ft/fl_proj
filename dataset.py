import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader

from torchvision import datasets

datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]


def get_mnist(data_path: str = './data'):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(data_path, train=True, download=True, transform=tr)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=tr)

    return train_set, test_set


def prepare_dataset(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1
                    ):

    train_set, test_set = get_mnist()

    # split train_set into 'num_partitions' train_sets
    num_images = len(train_set) // num_partitions

    partition_len = [num_images] * num_partitions
    train_sets = random_split(train_set, partition_len, torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    train_loaders = []
    val_loaders = []

    for train_set in train_sets:
        num_total = len(train_set)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(train_set, [num_train, num_val], torch.Generator().manual_seed(2023))

        train_loaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        val_loaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loaders, val_loaders, test_loader

