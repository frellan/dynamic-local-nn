from HebbianLayer import HebbianLayer
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden_layer = HebbianLayer(784, 128, learning_rate)
        self.output_layer = nn.Linear(128, 128)
        self.training = True

    def forward(self, features):
        y = self.hidden_layer(features)
        code = self.output_layer(y)
        return code

    def toggle_training(self, training):
        self.training = training
        self.hidden_layer.training = self.training
