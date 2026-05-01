import torch
import torch.nn as nn
from transformers import AutoModel

class MentalHealthMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", use_safetensors=True)
        hidden_size = self.phobert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # lấy token [CLS]
        logits = self.classifier(pooled)
        return logits