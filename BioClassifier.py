import torch
import torch.nn as nn
import torch.functional as F

class BioClassifier(nn.Module):
    # Wᵤᵢ is the unsupervised pretrained weight matrix of shape: (n_filters, img_sz)
    def __init__(self, Wᵤᵢ, out_features, n=4.5, β=.01):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ.transpose(0, 1) # (img_sz, n_filters)
        self.n = n
        self.β = β
        self.Sₐᵤ = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ): # vᵢ: (batch_sz, img_sz)
        Wᵤᵢvᵢ = torch.matmul(vᵢ, self.Wᵤᵢ)
        hᵤ = F.relu(Wᵤᵢvᵢ) ** self.n
        Sₐᵤhᵤ = self.Sₐᵤ(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤhᵤ)
        return cₐ
