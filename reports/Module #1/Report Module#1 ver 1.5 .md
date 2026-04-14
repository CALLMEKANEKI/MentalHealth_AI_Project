## Module 1: 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-10:  E:0.7 - H:0.3 

# ----------------Train---------------------------------------#
# Best model: Epoch 7 with Avg Acc: 0.7038, Epoch 9 with Avg Acc: 0.7107
Loading weights: 100%|████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 12469.25it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

--- Chiến lược Epoch 1: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [07:33<00:00,  1.48it/s, E=1.18, H=0.62, loss=0.8990]
📊 Kết quả Epoch 1: [Emotion Acc: 0.5466] | [Hate Acc: 0.6938]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6202

--- Chiến lược Epoch 2: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 2/10: 100%|████████████████████████████████████| 670/670 [09:44<00:00,  1.15it/s, E=0.68, H=1.08, loss=0.8821]
📊 Kết quả Epoch 2: [Emotion Acc: 0.6108] | [Hate Acc: 0.7547]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6828

--- Chiến lược Epoch 3: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:04<00:00,  1.38it/s, E=0.62, H=0.20, loss=0.4108]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5977] | [Hate Acc: 0.7778]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6877

--- Chiến lược Epoch 4: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:04<00:00,  1.38it/s, E=0.98, H=0.37, loss=0.7999]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5875] | [Hate Acc: 0.7669]

--- Chiến lược Epoch 5: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.39it/s, E=0.10, H=0.79, loss=0.3090]
📊 Kết quả Epoch 5: [Emotion Acc: 0.6501] | [Hate Acc: 0.7547]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7024

--- Chiến lược Epoch 6: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=0.04, H=0.98, loss=0.3214]
📊 Kết quả Epoch 6: [Emotion Acc: 0.6283] | [Hate Acc: 0.7696]

--- Chiến lược Epoch 7: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:04<00:00,  1.38it/s, E=0.04, H=0.03, loss=0.0379]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6312] | [Hate Acc: 0.7764]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7038

--- Chiến lược Epoch 8: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:04<00:00,  1.38it/s, E=0.12, H=0.47, loss=0.2244]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6210] | [Hate Acc: 0.7791]

--- Chiến lược Epoch 9: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.38it/s, E=0.05, H=0.03, loss=0.0424]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6356] | [Hate Acc: 0.7859]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7107

--- Chiến lược Epoch 10: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████| 670/670 [07:45<00:00,  1.44it/s, E=0.11, H=0.06, loss=0.0942]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6283] | [Hate Acc: 0.7913]



# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3424.69it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.42      0.38      0.39        40
     Disgust       0.57      0.61      0.59       132
   Enjoyment       0.70      0.71      0.71       193
        Fear       0.67      0.74      0.70        46
       Other       0.61      0.50      0.55       129
     Sadness       0.68      0.76      0.72       116
    Surprise       0.63      0.59      0.61        37

    accuracy                           0.64       693
   macro avg       0.61      0.61      0.61       693
weighted avg       0.63      0.64      0.63       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.87      0.86      0.87       600
   Offensive       0.72      0.74      0.73       496
        Hate       0.79      0.78      0.79       380

    accuracy                           0.80      1476
   macro avg       0.79      0.79      0.79      1476
weighted avg       0.80      0.80      0.80      1476





