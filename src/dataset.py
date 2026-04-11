from pydoc import text

import torch
from torch.utils.data import Dataset

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, cleaner, max_len=128): 
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.cleaner = cleaner
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        raw_text = str(self.texts[item])
        # Dọn dẹp văn bản trước khi đưa vào mô hình
        clean_text = self.cleaner.clean(raw_text)
        
        encoding = self.tokenizer(
            clean_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': clean_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }