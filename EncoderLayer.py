import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization
from  MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward



class EncoderLayer(nn.Module):
    def __init__(self,d_model:int,hidden_units:int, max_seq_len:int, attention_heads:int,dimension:list,eps:float, drop_prob:float ):
        super().__init__()
        self.attention = MultiHeadAttention(max_seq_len, d_model, attention_heads)
        self.norm1 = LayerNormalization(d_model,dimension,eps=eps)
        self.ff = PositionwiseFeedForward(d_model,hidden_units,drop_prob=drop_prob)
        self.norm2 =  LayerNormalization(d_model,dimension,eps=eps)
        self.dropout = nn.Dropout(p= drop_prob)
        
    def forward(self,x):
        x_res = x
        x = self.attention(x)
        x = self.dropout(x)
        x = self.norm1(x_res + x)
        x_res = x
        x = self.ff(x)
        x = self.dropout(x)
        x = self.norm2(x+x_res)
        return x
