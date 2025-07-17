from Decoder import Decoder
from Encoder import Encoder
from transformer_heads.Mask import Mask
import torch
from data_preparation.vocab_list import english_to_index, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
from data_preparation.SentenceEmbedding import SentenceEmbedding
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self,
                 d_model:int,
                 att_heads:int,
                 drop_prob:float,
                 batch_size:int,
                 ffn_hidden:int,
                 blocks:int,
                 eps:float,
                 dimension:list,
                 total_sentences:int,
                 max_sequence_length:int):
        super().__init__()
        self.eng_embedding = SentenceEmbedding(max_sequence_length=max_sequence_length,
                                           language_to_idx=english_to_index,
                                           START_TOKEN=START_TOKEN,
                                           END_TOKEN=END_TOKEN,
                                           PADDING_TOKEN=PADDING_TOKEN,
                                           d_model=d_model)
        self.kan_embedding = SentenceEmbedding(max_sequence_length=max_sequence_length,
                                           language_to_idx=kannada_to_index,
                                           START_TOKEN=START_TOKEN,
                                           END_TOKEN=END_TOKEN,
                                           PADDING_TOKEN=PADDING_TOKEN,
                                           d_model=d_model)
        self.encoder = Encoder(d_model=d_model,
                                hidden_units=ffn_hidden,
                                max_seq_len=max_sequence_length,
                                attention_heads=att_heads,
                                dimension=[dimension],
                                eps=eps,
                                drop_prob=drop_prob,
                                num_layers=blocks)
        
        self.decoder = Decoder(max_seqlen=max_sequence_length,
                                d_model=d_model,
                                attention_heads=att_heads,
                                dimension=[dimension],
                                eps=eps,
                                hidden_units=ffn_hidden,
                                drop_prob=drop_prob,
                                decoder_layers=blocks)
        
        self.mask = Mask(batch_size=batch_size, 
                        attention_head=att_heads, 
                        max_seq_len=max_sequence_length)
    
    def forward(self,dataset:tuple):
        english, kannada = dataset
        english = self.eng_embedding(english)
        english_enc = self.encoder(english)
        
        mask = self.mask()
        kannada = self.kan_embedding(kannada)
        kannada_dec = self.decoder(kannada, english_enc, mask)
        
        return kannada_dec