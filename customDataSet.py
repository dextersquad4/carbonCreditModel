import os
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, embeddings, values):
        self.values = values
        self.embeddings = embeddings
    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        value = self.values[idx]
        return embedding,value
    
