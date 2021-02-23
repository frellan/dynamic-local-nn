import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(128, 256)
        self.lin2 = nn.Linear(256, 784)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x