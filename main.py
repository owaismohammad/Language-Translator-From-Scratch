from Decoder import Decoder
from Encoder import Encoder
from transformer_heads.Mask import Mask
import torch

d_model = 3
num_heads = 1
drop_prob =  0.1
batch_size = 1
max_seq_len = 2
ffn_hidden = 5
num_layers = 5
eps = 1e-5
dimension = -1

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder(d_model=d_model,
                hidden_units=ffn_hidden,
                max_seq_len=max_seq_len,
                attention_heads=num_heads,
                dimension=[dimension],
                eps=eps,
                drop_prob=drop_prob,
                num_layers=num_layers).to(device)

raw_input = torch.tensor([[[1,2,3],
                       [4,5,6]],
                      
                      [[7,8,9],
                       [10,11,12]]], dtype=torch.float, device=device)

encoder_result= encoder(raw_input)



decoder = Decoder(max_seqlen=max_seq_len,
                  d_model=d_model,
                  attention_heads=num_heads,
                  dimension=[dimension],
                  eps=eps,
                  hidden_units=ffn_hidden,
                  drop_prob=drop_prob,
                  decoder_layers=num_layers).to(device)

mask_= Mask(batch_size=batch_size, 
            attention_head=num_heads, 
            max_seq_len=max_seq_len)

mask = mask_().to(device)
decoder_input = raw_input
x = decoder(decoder_input, encoder_result, mask)

print(f"encoder result : {encoder_result}")
print(f"decoder output : {x}")
