import torch
from torch.utils.data import Dataset
import pandas as pd

class MultiTaskDataset(Dataset):
    def __init__(self, texts, emotion_labels, hate_labels, tokenizer, max_len):
        """
        texts: array-like of strings (có thể chứa NaN)
        emotion_labels: list of int (hoặc -100)
        hate_labels: list of int (hoặc -100)
        tokenizer: PhoBERT tokenizer
        max_len: int
        """
        self.emotion_labels = emotion_labels
        self.hate_labels = hate_labels
        
        # Xử lý NaN và đảm bảo mọi giá trị là string
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()  # nếu là pandas Series/array
        texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        # Tokenize toàn bộ dữ liệu một lần
        print("⏳ Tokenizing dataset... (chỉ một lần)")
        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None
        )
        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']
        print("✅ Tokenization hoàn tất!")

    def __len__(self):
        return len(self.emotion_labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotion_labels[idx], dtype=torch.long),
            'hate_labels': torch.tensor(self.hate_labels[idx], dtype=torch.long)
        }