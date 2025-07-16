import torch.nn as nn
import torch
class Mask(nn.Module):
    def __init__(self, batch_size:int, attention_head:int, max_seq_len:int):
        super().__init__()
        self.batch = batch_size
        self.ah = attention_head
        self.maxseq = max_seq_len

    def forward(self):
        z = torch.zeros((self.batch,self.ah,self.maxseq,self.maxseq))
        z = torch.fill(z,value=torch.tensor(float('-inf')))
        z = torch.triu(z,diagonal=1)
        return z
