import torch
import torch.nn as nn

class UnsupervisedLayer(nn.Module):
    def __init__(self, in_features, out_features, learning_rate):
        super(UnsupervisedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.rand(out_features, in_features)

        self.learning_rate = learning_rate
        self.learn = False

    def output(self, input):
        return input.matmul(self.weights.t())

    def forward(self, input):
        y = self.output(input)
        if self.learn:
            self.update_weights_krotov(input)
        return y

    def update_weights_krotov(self, input):
        precision = 1e-30
        anti_hebbian_learning_strength = 0.4
        lebesgue_norm = 2.0
        rank = 2
        batch_size = input.shape[0]

        mini_batch = torch.transpose(input, 0, 1)
        sign = torch.sign(self.weights)
        W = sign * torch.abs(self.weights) ** (lebesgue_norm - 1)
        tot_input = torch.mm(W, mini_batch)

        y = torch.argsort(tot_input, dim=0)
        yl = torch.zeros((self.out_features, batch_size), dtype = torch.float)
        yl[y[self.out_features - 1,:], torch.arange(batch_size)] = 1.0
        yl[y[self.out_features - rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength

        xx = torch.sum(yl * tot_input, 1)
        xx = xx.unsqueeze(1)
        xx = xx.repeat(1, self.in_features)
        ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * self.weights

        nc = torch.max(torch.abs(ds))
        if nc < precision: nc = precision
        self.weights += self.learning_rate * (ds / nc)

    def reset_weights(self):
        self.weights = torch.rand(self.out_features, self.in_features)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )