import torch
import torch.nn as nn

class BioLoss(nn.Module):
    def __init__(self, out_features, device, m=6):
        super().__init__()
        self.device = device
        self.out_features = out_features
        self.m = m

    def forward(self, cₐ, tₐ): 
        tₐ_ohe = torch.eye(self.out_features, dtype=torch.float, device=self.device)[tₐ]
        tₐ_ohe[tₐ_ohe==0] = -1.
        loss = (cₐ - tₐ_ohe).abs() ** self.m
        return loss.sum()
