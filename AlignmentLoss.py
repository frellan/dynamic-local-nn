import torch
import torch.nn as nn

class AlignmentLoss(nn.Module):
    def __init__():
        super().__init__()

    def forward(output, target): 
        size = output.shape[0]
        k1 = torch.tril(output @ output.t(), diag = -1)
        k2 = torch.zeros((size,size))
        for i in range(size):
            for j in range(i):
                k2[i, j] = target[i] == target[j]
        return -(torch.sum(k1 * k2) / (torch.sum(k2) * torch.norm(k1)))