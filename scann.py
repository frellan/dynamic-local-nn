import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datasets import mnist

from SCANN.SCANN import SCANN

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_loader, test_loader = mnist.get_loaders(train_bz=5000, test_bz=128, normalize_0_1=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784

system = SCANN(input_size, 2000, device).to(device)

system.learn_unsupervised_weights(train_loader, 200)
system.run_test(train_loader, test_loader, epochs=10)
