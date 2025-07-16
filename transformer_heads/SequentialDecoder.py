import torch
import torch.nn as nn

class SequentialDecoder(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, *inputs):
        decoder_input,encoder_result, mask = inputs
        for layer in self.layers:
            decoder_input = layer(decoder_input,encoder_result,mask)
        return decoder_input
    
