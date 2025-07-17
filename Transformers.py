from Decoder import Decoder
from Encoder import Encoder
from data_preparation.get_data import Clean
from transformer_heads.Mask import Mask
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_preparation.vocab_list import english_to_index, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
from data_preparation.SentenceEmbedding import SentenceEmbedding

d_model = 512
num_heads = 8
drop_prob =  0.1
batch_size = 32
max_seq_len = 200
ffn_hidden = 2084
num_layers = 1
eps = 1e-5
dimension = -1
total_sentences = 100000

device = "cuda" if torch.cuda.is_available() else "cpu"

english_txt = Path('data_preparation/train/english.txt')
kannada_txt = Path('data_preparation/train/kannada.txt')


txt_clean = Clean(english_txt=english_txt, 
                  kannada_txt=kannada_txt, 
                  max_sequence_length=max_seq_len,
                  TOTAL_SENTENCES= total_sentences )

dataset = txt_clean.clean()

dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

eng, kan = next(iter(dataloader))

eng_embed = SentenceEmbedding(max_sequence_length=200,
                          d_model= 512,
                          language_to_idx=english_to_index,
                          START_TOKEN=START_TOKEN,
                          END_TOKEN=END_TOKEN,
                          PADDING_TOKEN=PADDING_TOKEN)
kan_embed = SentenceEmbedding(max_sequence_length=200,
                          d_model= 512,
                          language_to_idx=kannada_to_index,
                          START_TOKEN=START_TOKEN,
                          END_TOKEN=END_TOKEN,
                          PADDING_TOKEN=PADDING_TOKEN)

english_embeddings = eng_embed(eng)
kannada_embeddings = kan_embed(kan)

# encoder = Encoder(d_model=d_model,
#                 hidden_units=ffn_hidden,
#                 max_seq_len=max_seq_len,
#                 attention_heads=num_heads,
#                 dimension=[dimension],
#                 eps=eps,
#                 drop_prob=drop_prob,
#                 num_layers=num_layers).to(device)


# encoder_result= encoder(english_embeddings)



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

# x = decoder(kannada_embeddings, encoder_result, mask)

print(f"encoder result : {encoder_result.shape}")
# print(f"decoder output : {x.shape}")
