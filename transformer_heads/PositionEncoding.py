import torch.nn as nn
import torch

class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len:int, d_model:int):
        super().__init__()
        self.maxseq = max_seq_len
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(0, self.maxseq).reshape(self.maxseq, 1)
        even_denom = torch.pow(10000, torch.arange(0,self.d_model,2)/self.d_model )
        odd_denom = torch.pow(10000, torch.arange(1,self.d_model,2)/self.d_model ) 
        z = x
        z[...,0::2] = torch.sin(pos/even_denom)
        z[...,1::2] = torch.cos(pos/odd_denom)
        return z

# input= torch.rand((3,200,512), dtype=torch.float)
# model = PositionEncoding(max_seq_len=200,d_model=512)
# y = model(input)
# print(f"y={y}")
