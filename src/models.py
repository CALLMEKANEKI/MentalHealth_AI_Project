import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PhoBERTForSentiment(nn.Module):
    def __init__(self, num_labels):
        super(PhoBERTForSentiment, self).__init__()
        model_name = "vinai/phobert-base"
        # Thêm tham số use_safetensors=True ở đây
        self.phobert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        # Tầng Dropout để chống học vẹt (Overfitting)
        self.dropout = nn.Dropout(0.3)
        # Tầng Tuyến tính để phân loại (đầu ra là số lượng nhãn của bạn)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Đưa dữ liệu qua PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy vector đại diện của câu (từ token [CLS] - thường là đầu ra đầu tiên)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)