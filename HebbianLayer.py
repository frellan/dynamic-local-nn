import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLayer(nn.Module):
    def __init__(self, in_features, out_features, learning_rate):
        super(HebbianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.rand(out_features, in_features)
        self.bias = torch.zeros(out_features)

        self.learning_rate = learning_rate
        self.learning = True

    def output(self, input):
        # return torch.tanh(F.linear(input, self.weights, self.bias))
        return F.linear(input, self.weights, self.bias)

    def forward(self, input):
        y = self.output(input)
        if self.learning:
            self.update_weights(input)
        return y

    def update_weights(self, input):
        count = 0
        for x_input in input:
            # if count >= 1:
            #     continue
            x = x_input.reshape(1, len(x_input))
            y = self.output(x_input)
            y_length = y.shape[0]
            y = y.reshape(1, y_length)
            y_t = y.reshape(y_length, 1)

            d_w = self.learning_rate * ((x * y_t) - (torch.tril(y * y_t) @ self.weights))

            # print("x: ", x)
            # print("x SHAPE: ", x.shape)
            # print("self.weights: ", self.weights)
            # print("self.weights SHAPE: ", self.weights.shape)
            # print("y: ", y)
            # print("y SHAPE: ", y.shape)
            # print("y_t: ", y_t)
            # print("d_w: ", d_w)
            # print("d_w SHAPE: ", d_w.shape)

            self.weights += d_w
            count += 1

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )
