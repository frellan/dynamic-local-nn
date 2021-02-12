from HebbianLayer import HebbianLayer
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden_layer = HebbianLayer(784, 128, learning_rate)
        # self.output_layer = HebbianLayer(128, 128, learning_rate)
        self.output_layer = nn.Linear(128, 128)

    def forward(self, features):
        y = self.hidden_layer(features)
        # y = F.relu(y)
        code = self.output_layer(y)
        return code
        # output = F.relu(code)
        # return output
