import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 784)

    def forward(self, features):
        y = self.hidden_layer(features)
        y = F.relu(y)
        y = self.output_layer(y)
        output = F.relu(y)
        return output