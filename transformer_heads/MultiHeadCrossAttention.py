import torch
import torch.nn as nn

class MultiHeadAttentionCrossAttention(nn.Module):
    def __init__(self, max_seqlen:int, d_model:int, attention_heads:int):
        super().__init__()
        self.max_seq = max_seqlen
        self.d_model = d_model
        self.ah = attention_heads

        self.kv = nn.Linear(d_model, d_model*2)
        self.q = nn.Linear(d_model, d_model)

    @staticmethod
    def scaled_dot_product(q,k,v):

        d_k = torch.tensor(k.shape[-1], dtype=torch.float)
        attention = (q @ k.transpose(-2,-1)) / torch.sqrt(d_k)
        attention = torch.softmax(attention, dim =-1)
        scaled = attention @ v
        return scaled

    def forward(self,y,x): #first (y)=> decoder input , second (x) => encoder result
        batch_size, seq_len, d_model = x.size() # 30x200x512
        
        kv = self.kv(x)
        q = self.q(y)
        
        ad = d_model // self.ah
        kv = kv.reshape(batch_size,seq_len, self.ah, ad*2 ) # 30x8x200x128
        kv = kv.permute(0,2,1,3)
        k,v = kv.chunk(chunks=2, dim=3)
        
        q = q.reshape(y.shape[0], y.shape[1], self.ah, ad ) # 30x8x200x64
        q = q.permute(0,2,1,3)

        values = MultiHeadAttentionCrossAttention.scaled_dot_product(q,k,v)
        batch, head, seq, dim = values.size()
        values = values.reshape(batch, seq, dim*head)

        return values 


input = torch.tensor([[[1,2,3],
                       [4,5,6]],
                      
                      [[7,8,9],
                       [10,11,12]]], dtype=torch.float)
print(input.size())
# encoder = Encoder.Encoder(3,5,2,1,[-1],1e-5,0.1,1)
# z = encoder(input)
# model = MultiHeadAttentionCrossAttention(2,3,1)
# y = model(z,input)
# print(f"y = {y}")