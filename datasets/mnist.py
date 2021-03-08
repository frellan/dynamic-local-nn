import torch
import torchvision.datasets
from torch.utils.data import DataLoader, TensorDataset

def get_loaders(flatten=True, train_bz=1, test_bz=128, fmnist=False, normalize_0_1=False):
    if fmnist:
        mnist = torchvision.datasets.FashionMNIST("~/datasets", train=True, download=True)
        mnist_test = torchvision.datasets.FashionMNIST("~/datasets", train=False, download=True)
    else:
        mnist = torchvision.datasets.MNIST("~/datasets", train=True, download=True)
        mnist_test = torchvision.datasets.MNIST("~/datasets", train=False, download=True)

    # Normalize
    dtype = torch.get_default_dtype()
    Xt, Xv = mnist.data.to(dtype), mnist_test.data.to(dtype)
    if normalize_0_1:
        X_min, X_max = Xt.min(), Xt.max()
        Xt, Xv = (Xt - X_min) / (X_max - X_min), (Xv - X_min) / (X_max - X_min)
    else:
        X_mean, X_std = Xt.mean(), Xt.std()
        Xt, Xv = (Xt - X_mean) / X_std, (Xv - X_mean) / X_std

    yt, yv = mnist.targets, mnist_test.targets

    if flatten:
        Xt, Xv = Xt.view(Xt.shape[0], -1), Xv.view(Xv.shape[0], -1)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=train_bz, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xv, yv), batch_size=test_bz)

    return train_loader, test_loader
