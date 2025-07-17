from anyio import Path
from Decoder import Decoder
from Encoder import Encoder
from Transformers import Transformer
from data_preparation.get_data import Clean
from torch.utils.data import DataLoader
import torch

d_model = 3
att_heads = 1
drop_prob =  0.1
batch_size = 1
max_seq_len = 2
ffn_hidden = 5
blocks = 5
eps = 1e-5
dimension = -1
total_sentences = 4
EPOCHS=10

transformer = Transformer(d_model=d_model,
                          att_heads=att_heads,
                          drop_prob=drop_prob,
                          batch_size=batch_size,
                          ffn_hidden=ffn_hidden,
                          blocks=blocks,
                          eps=eps,
                          dimension=[dimension],
                          total_sentences=total_sentences,
                          max_sequence_length=max_seq_len)

english_txt = Path('data_preparation/train/english.txt')
kannada_txt = Path('data_preparation/train/kannada.txt')

model = Clean(english_txt=english_txt, kannada_txt=kannada_txt,TOTAL_SENTENCES=total_sentences, max_sequence_length=max_seq_len)

dataset = model.clean()

loader = DataLoader(dataset,batch_size=32, drop_last=True)


for epoch in range(EPOCHS):
    for batch in loader:
        y = transformer(batch)
        











# d_model = 3
# num_heads = 1
# drop_prob =  0.1
# batch_size = 1
# max_seq_len = 2
# ffn_hidden = 5
# num_layers = 5
# eps = 1e-5
# dimension = -1

# device = "cuda" if torch.cuda.is_available() else "cpu"
# encoder = Encoder(d_model=d_model,
#                 hidden_units=ffn_hidden,
#                 max_seq_len=max_seq_len,
#                 attention_heads=num_heads,
#                 dimension=[dimension],
#                 eps=eps,
#                 drop_prob=drop_prob,
#                 num_layers=num_layers).to(device)

# raw_input = torch.tensor([[[1,2,3],
#                        [4,5,6]],
                      
#                       [[7,8,9],
#                        [10,11,12]]], dtype=torch.float, device=device)

# encoder_result= encoder(raw_input)



# decoder = Decoder(max_seqlen=max_seq_len,
#                   d_model=d_model,
#                   attention_heads=num_heads,
#                   dimension=[dimension],
#                   eps=eps,
#                   hidden_units=ffn_hidden,
#                   drop_prob=drop_prob,
#                   decoder_layers=num_layers).to(device)

# mask_= Mask(batch_size=batch_size, 
#             attention_head=num_heads, 
#             max_seq_len=max_seq_len)

# mask = mask_().to(device)
# decoder_input = raw_input
# x = decoder(decoder_input, encoder_result, mask)

# print(f"encoder result : {encoder_result}")
# print(f"decoder output : {x}")
