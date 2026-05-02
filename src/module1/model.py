# =============================================================================
# Module 1 - Model (Ver 2.6)
# =============================================================================
# Thay đổi so với ver 2.5:
#   - Đổi backbone từ vinai/phobert-base → vinai/phobert-base-v2
#     Lý do: v2 train trên corpus lớn hơn, tốt hơn cho social media VN
#   - Thêm Label Smoothing vào Emotion head (label_smoothing_eps)
#     Lý do: tránh model quá confident vào nhãn phổ biến (joy, sadness)
#             giúp các nhãn hiếm (disapproval, realization) học tốt hơn
#   - Giữ nguyên kiến trúc head (768→384→256→28 và 768→256→3)
# =============================================================================

import torch
import torch.nn as nn
from transformers import AutoModel
from module1.dataset import NUM_VIGO_LABELS

# Đổi sang phobert-base-v2
PHOBERT_MODEL = "vinai/phobert-base-v2"


class PhoBERTMultiTask(nn.Module):
    """
    Multi-task model:
    - Task 1 Emotion:     Multi-label, 28 nhãn ViGoEmotions → BCEWithLogitsLoss
    - Task 2 Hate Speech: Single-label, 3 nhãn              → CrossEntropyLoss
    """

    def __init__(self, num_hate_labels=3, num_emotion_labels=NUM_VIGO_LABELS, dropout=0.3):
        super(PhoBERTMultiTask, self).__init__()

        # Ver 2.6: Dùng phobert-base-v2 thay vì phobert-base
        self.phobert = AutoModel.from_pretrained(
            PHOBERT_MODEL, use_safetensors=True
        )
        hidden_size       = self.phobert.config.hidden_size  # 768
        intermediate_size = 256

        # --- Emotion Head: Multi-label 28 nhãn ---
        # Không dùng sigmoid ở đây — BCEWithLogitsLoss tự xử lý
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),        # 768 → 384
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, intermediate_size),  # 384 → 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_emotion_labels) # 256 → 28
        )

        # --- Hate Speech Head: Single-label 3 nhãn ---
        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),   # 768 → 256
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, num_hate_labels)  # 256 → 3
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling — tốt hơn cho social media text ngắn
        last_hidden  = outputs.last_hidden_state
        mask_exp     = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb      = torch.sum(last_hidden * mask_exp, 1)
        sum_mask     = torch.clamp(mask_exp.sum(1), min=1e-9)
        pooled       = sum_emb / sum_mask  # [batch, 768]

        emo_logits  = self.emotion_head(pooled)  # [batch, 28]
        hate_logits = self.hate_head(pooled)     # [batch, 3]

        return emo_logits, hate_logits