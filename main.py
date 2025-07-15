from Encoder import Encoder
import torch

d_model = 3
num_heads = 1
drop_prob =  0.1
batch_size = 1
max_seq_len = 4
ffn_hidden = 5
num_layers = 5
eps = 1e-5
dimension = -1

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Encoder(d_model=d_model,
                hidden_units=ffn_hidden,
                max_seq_len=max_seq_len,
                attention_heads=num_heads,
                dimension=[dimension],
                eps=eps,
                drop_prob=drop_prob,
                num_layers=num_layers).to(device)

input = torch.tensor([[[1,2,3],
                       [4,5,6]],
                      
                      [[7,8,9],
                       [10,11,12]]], dtype=torch.float, device=device)

y= model(input)
print(y)