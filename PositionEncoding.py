import torch.nn as nn
import torch

class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len:int, d_model:int):
        super().__init__()
        self.maxseq = max_seq_len
        self.d_model = d_model

    def forward(self, x):
        batch, seq, dim = x.size()
        pos = torch.arange(0, self.maxseq).reshape(self.maxseq, 1)
        print(f"pos = {pos}")
        even_denom = torch.pow(10000, torch.arange(0,self.d_model,2)/self.d_model )
        print(f"even deno = {even_denom}")
        odd_denom = torch.pow(10000, torch.arange(1,self.d_model,2)/self.d_model )
        print(f"odd denom = {odd_denom}")
        print(f"pos/even denom = {pos/even_denom}")
        print(f"pos/odd denom = {pos/odd_denom}")
        print(f"{(pos/even_denom).unsqueeze(1)}")
        z = x
        z[...,0::2] = (pos/even_denom).unsqueeze(1)
        z[...,1::2] = (pos/odd_denom).unsqueeze(1)
        return z

input= torch.tensor([[[1,2,3],
                       [4,5,6]],

                      [[7,8,9],
                       [10,11,12]]], dtype=torch.float)
model = PositionEncoding(2,3)
y = model(input)
y
