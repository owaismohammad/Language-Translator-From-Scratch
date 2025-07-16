import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,d_model:int, dimension:list, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dimension = dimension
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        mean = torch.mean(x, dim = self.dimension, keepdim=True)
        var = torch.var(x, dim = self.dimension, keepdim=True)
        layer_norm = (x - mean) / torch.sqrt(var + self.eps)
        norm = self.gamma * layer_norm + self.bias
        return norm

