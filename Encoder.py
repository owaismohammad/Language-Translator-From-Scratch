import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model: int, hidden_units: int, max_seq_len: int, attention_heads: int,dimension: list,eps: float, drop_prob: float, num_layers ):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model=d_model,
                                                   hidden_units=hidden_units,
                                                   max_seq_len=max_seq_len,
                                                   attention_heads=attention_heads,
                                                   dimension=dimension,
                                                   eps=eps,
                                                   drop_prob=drop_prob)
                                      for _ in range(num_layers)])
    def forward(self,x):
        return self.layers(x)
        