import os
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from data_preparation.SentenceEmbedding import SentenceEmbedding
from data_preparation.vocab_list import END_TOKEN, PADDING_TOKEN, START_TOKEN, kannada_vocabulary,english_vocabulary
from data_preparation.helper_functions import TextDataset, check_sentences


class Clean():

    def __init__(self,
                 english_txt:pathlib.Path,
                 kannada_txt:pathlib.Path,
                 max_sequence_length:int,
                 TOTAL_SENTENCES:int):
        self.english_txt = english_txt
        self.kannada_txt = kannada_txt
        self.total_sent = TOTAL_SENTENCES
        self.max_sequence_length = max_sequence_length
        
    @staticmethod    
    def read_txt():
        with open(english_txt, "r") as file:
            english_sentences = file.readlines()
    
        with open(kannada_txt, "r") as file:
            kannada_sentences = file.readlines()
        return english_sentences, kannada_sentences
    
    def clean(self):
        eng_sent, kan_sent = Clean.read_txt()
        english_sentences, kannada_sentences = eng_sent[:self.total_sent], kan_sent[:self.total_sent]
        eng_sent, kan_sent = [x.rstrip() for x in english_sentences], [x.rstrip() for x in kannada_sentences]
        english_to_index = {k:v for v,k in enumerate(english_vocabulary)}
        kannada_to_index = {k:v for v,k in enumerate(kannada_vocabulary)}

        valid_indices=[]

        valid_indices = check_sentences(TOTAL_SENTENCES=self.total_sent,
                                        kannada_lang=kan_sent,
                                        eng_language=eng_sent,
                                        eng_to_index=english_to_index,
                                        kannada_to_index=kannada_to_index,
                                        max_seq_len=self.max_sequence_length)

        english = [eng_sent[i] for i in valid_indices]
        kannada = [kan_sent[i] for i in valid_indices] 
        dataset = TextDataset(english, kannada)

        return dataset
    
    
    
english_to_index = {k:v for v,k in enumerate(english_vocabulary)}
kannada_to_index = {k:v for v,k in enumerate(kannada_vocabulary)}


english_txt = Path('data_preparation/train/english.txt')
kannada_txt = Path('data_preparation/train/kannada.txt')

model = Clean(english_txt=english_txt, kannada_txt=kannada_txt,TOTAL_SENTENCES=1000, max_sequence_length=200)

dataset = model.clean()

loader = DataLoader(dataset,batch_size=32, drop_last=True)


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
english_embeddings=[]
kannada_embeddings=[]


for batch in loader:
    eng, kan = batch
    english_embeddings.append( eng_embed(eng))
    # kannada_embeddings.append(kan_embed(kan))
    
print(english_embeddings.shape)    
    
