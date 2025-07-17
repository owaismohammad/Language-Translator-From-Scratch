from operator import length_hint
from torch.utils.data import Dataset

def is_valid_token(lang_to_index, sentence):
    for token in sentence:
        if token not in lang_to_index:
            return False
    return True

def length(sentence, max_seq_len):
    if len(sentence) > max_seq_len:
        return False
    return True

def check_sentences(TOTAL_SENTENCES,kannada_lang, eng_language, eng_to_index, kannada_to_index, max_seq_len):
    valid_indices=[]
    for i in range(TOTAL_SENTENCES):
        if is_valid_token(eng_to_index, eng_language[i]) and is_valid_token(kannada_to_index,kannada_lang[i]) and length(eng_language[i], max_seq_len) and length(kannada_lang[i], max_seq_len):
            valid_indices.append(i)
            
    return valid_indices

class TextDataset(Dataset):
    def __init__(self, english, kannada):
        super().__init__()
        self.english = english
        self.kannada = kannada
        
    def __len__(self):
        return len(self.english)
    
    def __getitem__(self, index):
        return self.english[index], self.kannada[index]