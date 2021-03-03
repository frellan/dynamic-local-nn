from SCANN.UnsupervisedLayer import UnsupervisedLayer
import torch.nn as nn

class UnsupervisedModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.learning_rate = 2e-2
        self.layers = [
            UnsupervisedLayer(input_size, input_size // 4, self.learning_rate)
        ]
        self.input_size = input_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_and_learn(self, x):
        for layer in self.layers:
            layer.learn = True
            x = layer(x)
            layer.learn = False
        return x

    def get_weights(self):
        return self.layers[0].weights
