import torch.nn as nn
import torch
import Encoder
from transformer_heads.MultiHeadAttention import MultiHeadAttention
from transformer_heads.LayerNormalization import LayerNormalization
from transformer_heads.MultiHeadCrossAttention import MultiHeadAttentionCrossAttention
from transformer_heads.PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, max_seqlen:int, d_model:int, attention_heads:int, dimension:list, eps:float, hidden_units:int, drop_prob:float):
        super().__init__()
        self.maskmultihead = MultiHeadAttention(max_seqlen=max_seqlen,
                                                d_model=d_model,
                                                attention_heads=attention_heads)
        self.norm1 = LayerNormalization(d_model=d_model,
                                        dimension= dimension,
                                        eps = eps)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        self.crossatt = MultiHeadAttentionCrossAttention(max_seqlen=max_seqlen, d_model=d_model, attention_heads=attention_heads)
        self.norm2 = LayerNormalization(d_model=d_model,
                                        dimension= dimension,
                                        eps = eps)
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.ff = PositionwiseFeedForward(d_model=d_model,
                                          hidden_unit=hidden_units,
                                          drop_prob=drop_prob)
        self.norm3 = LayerNormalization(d_model=d_model,
                                        dimension= dimension,
                                        eps = eps)
        self.dropout3 = nn.Dropout(p = drop_prob)
        
    def forward(self,x,y, mask):
        x_res = x
        x = self.maskmultihead(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + x_res)
        
        x_res = x
        x = self.crossatt(x,y)
        x = self.dropout2(x)
        x = self.norm2(x+x_res)        
        
        x_res = x
        x = self.ff(x)
        x = self.dropout3(x)
        x = self.norm3(x+x_res)
        return x