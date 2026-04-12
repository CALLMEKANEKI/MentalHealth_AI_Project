import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERTMultiTask(nn.Module):
    def __init__(self, num_emotion_labels=7, num_hate_labels=3):
        super(PhoBERTMultiTask, self).__init__()
        # Load phần thân PhoBERT
        self.phobert = AutoModel.from_pretrained(
            "vinai/phobert-base", 
            use_safetensors=True  # Sử dụng định dạng an toàn để tránh lỗ hổng bảo mật
        )
        
        # Dropout để chống Overfitting (học vẹt)
        self.dropout = nn.Dropout(0.1)
        
        # Nhánh 1: Dự đoán cảm xúc (Emotion)
        self.emotion_head = nn.Linear(768, num_emotion_labels)
        
        # Nhánh 2: Dự đoán độc hại (Hate Speech)
        self.hate_head = nn.Linear(768, num_hate_labels)

    def forward(self, input_ids, attention_mask):
        # Đưa dữ liệu qua PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy vector đại diện của toàn bộ câu (CLS token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Đẩy qua 2 đầu độc lập
        emotion_logits = self.emotion_head(pooled_output)
        hate_logits = self.hate_head(pooled_output)
        
        return emotion_logits, hate_logits