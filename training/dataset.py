import torch
from torch.utils.data import Dataset
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConversationDataset(Dataset):
    """Dataset for conversation data"""
    def __init__(self, csv_path, tokenizer, max_length=50):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} conversation pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Encode input and response
        input_text = str(row['input'])
        response_text = str(row['response'])
        
        input_ids = self.tokenizer.encode(input_text, self.max_length)
        target_ids = self.tokenizer.encode(response_text, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }
