import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HebbianLayer(nn.Module):
    def __init__(self, in_features, out_features, learning_rate):
        super(HebbianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.learning_rate = learning_rate
        self.learning = True

    def activation(self, input):
        return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        y = self.activation(input)
        if self.learning:
            self.update_weight(input)
        return y

    def update_weight(self, input):
        with torch.no_grad():
            for x_input in input:
                x = x_input.reshape(1, len(x_input))
                y = self.activation(x_input)
                y_length = len(y)
                y = y.reshape(1, y_length)
                y_t = y.reshape(y_length, 1)
                d_w = self.learning_rate * ((x * y_t) - torch.tril(y * y_t, diagonal=-1) @ self.weight)
                self.weight += d_w

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )
