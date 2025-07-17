from h11 import Data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preparation import SentenceEmbedding
from data_preparation.helper_functions import TextDataset

class DataLoading(nn.Module):
    def __init__(self, batch_size:int):
        self.batch_size = batch_size
    
    def forward(self, dataset : TextDataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        for batch in dataloader:
            # eng_embeddings = SentenceEmbedding()
            pass        