from HebbianLayer import HebbianLayer
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, learning_rate=2e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.hebb1 = HebbianLayer(784, 256, learning_rate)
        self.hebb2 = HebbianLayer(256, 128, learning_rate)
        self.training = True

    def forward(self, x):
        x = self.hebb1(x)
        x = self.hebb2(x)
        return x

    def toggle_training(self, training):
        self.training = training
        self.hebb1.training = self.training
        self.hebb2.training = self.training
