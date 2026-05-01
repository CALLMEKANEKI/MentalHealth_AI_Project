import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MentalHealthMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.labels = np.array(labels, dtype=np.float32)  # shape (n_samples, n_labels)
        safe_texts = [str(t) if pd.notna(t) else " " for t in texts]
        encodings = tokenizer(
            safe_texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors=None
        )
        self.input_ids = np.array(encodings['input_ids'], dtype=np.int64)
        self.attention_masks = np.array(encodings['attention_mask'], dtype=np.int64)

        vocab_size = tokenizer.vocab_size
        for i, ids in enumerate(self.input_ids):
            if np.any(ids >= vocab_size) or np.any(ids < 0):
                ids[ids >= vocab_size] = tokenizer.pad_token_id
                ids[ids < 0] = tokenizer.pad_token_id
                self.input_ids[i] = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }