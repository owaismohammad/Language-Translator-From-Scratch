
import torch 
import torch.nn as nn

from transformer_heads.PositionEncoding import PositionEncoding
class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length:int,
                        d_model:int,
                        language_to_idx:dict,
                        START_TOKEN:str,
                        END_TOKEN:str,
                        PADDING_TOKEN:str):
        super().__init__()
        self.vocab_size = len(language_to_idx)
        self.max_seq = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_idx = language_to_idx
        self.Position_Encoder = PositionEncoding(self.max_seq, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN


    def batch_tokenize(self, batch, start_token = True, end_token = True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indices = [self.language_to_idx[token] for token in sentence]

            if start_token==True:
                sentence_word_indices.insert(0, self.language_to_idx[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_idx[self.END_TOKEN])
            for _ in range(len(sentence_word_indices),self.max_seq) :
                sentence_word_indices.append(self.language_to_idx[self.PADDING_TOKEN])
                
            if len(sentence_word_indices) > self.max_seq:
                sentence_word_indices = sentence_word_indices[:self.max_seq]
                
            # print(len(sentence_word_indices))    
            return sentence_word_indices


        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append( tokenize(batch[sentence_num], start_token= start_token, end_token=end_token))

        tokenized = torch.tensor(tokenized)  
        return tokenized


    def forward(self,x, start_token= True, end_token = True):
        x = self.batch_tokenize(x, start_token, end_token)
        x= self.embedding(x)
        pos = self.Position_Encoder(x)
        x = self.dropout(x + pos)
        return x

