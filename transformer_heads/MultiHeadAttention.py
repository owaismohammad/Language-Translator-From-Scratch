import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, max_seqlen:int, d_model:int, attention_heads:int):
        super().__init__()
        self.max_seq = max_seqlen
        self.d_model = d_model
        self.ah = attention_heads

        self.qkv = nn.Linear(d_model, d_model*3)

    @staticmethod
    def scaled_dot_product(q,k,v,mask=None):

        d_k = torch.tensor(k.shape[-1], dtype=torch.float)
        attention = (q @ k.transpose(-2,-1)) / torch.sqrt(d_k)
        if mask is not None:
            attention+=mask
        attention = torch.softmax(attention, dim =-1)
        scaled = attention @ v
        return scaled

    def forward(self,x, mask=None):
        qkv = self.qkv(x)
        ad = self.d_model // self.ah
        qkv = qkv.reshape(x.shape[0],x.shape[1], self.ah, ad*3 )
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(chunks=3, dim=3)
        values= MultiHeadAttention.scaled_dot_product(q,k,v, mask=mask)
        batch, head, seq, dim = values.size()
        values = values.reshape(batch, seq, dim*head)

        return values 
