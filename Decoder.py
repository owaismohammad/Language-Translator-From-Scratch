import torch
import torch.nn as nn
from DecoderLayer import DecoderLayer
from transformer_heads.SequentialDecoder import SequentialDecoder

class Decoder(nn.Module):
    def __init__(self,max_seqlen: int,
                    d_model: int,
                    attention_heads: int,
                    dimension: list,
                    eps: float,
                    hidden_units: int,
                    drop_prob: float,
                    decoder_layers:int):
        
        super().__init__()
        self.decoder = SequentialDecoder(*[DecoderLayer(max_seqlen=max_seqlen,
                                                    d_model=d_model,
                                                    attention_heads=attention_heads,
                                                    dimension=dimension,
                                                    eps= eps,
                                                    hidden_units=hidden_units,
                                                    drop_prob=drop_prob,
                                                    ) for _ in range(decoder_layers)])
        
    def forward(self,decoder_input,encoder_result,mask):
        return self.decoder(decoder_input,encoder_result,mask)