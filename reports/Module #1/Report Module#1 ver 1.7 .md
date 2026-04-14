## Module 1: Chỉnh sửa Batch Loss và Head Architecture
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-10:  E:0.7 - H:0.3 
# ----------------Train---------------------------------------#
# Best model: Epoch 8 with Avg Acc: 0.7117
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6840.18it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |


--- Chiến lược Epoch 1: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [07:37<00:00,  1.47it/s, E=1.15, H=0.82, loss=0.9878]
📊 Kết quả Epoch 1: [Emotion Acc: 0.5394] | [Hate Acc: 0.6369]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5881

--- Chiến lược Epoch 2: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:39<00:00,  1.46it/s, E=0.77, H=0.65, loss=0.7062]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5831] | [Hate Acc: 0.7358]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6594

--- Chiến lược Epoch 3: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:40<00:00,  1.46it/s, E=0.38, H=0.71, loss=0.5448]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5466] | [Hate Acc: 0.7236]

--- Chiến lược Epoch 4: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=1.38, H=0.36, loss=1.0710]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5729] | [Hate Acc: 0.7344]

--- Chiến lược Epoch 5: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=1.06, H=0.10, loss=0.7739]
📊 Kết quả Epoch 5: [Emotion Acc: 0.6093] | [Hate Acc: 0.7724]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6908

--- Chiến lược Epoch 6: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=0.55, H=0.95, loss=0.6719]
📊 Kết quả Epoch 6: [Emotion Acc: 0.6079] | [Hate Acc: 0.7696]

--- Chiến lược Epoch 7: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:37<00:00,  1.47it/s, E=0.25, H=0.40, loss=0.2979]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6312] | [Hate Acc: 0.7873]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7092

--- Chiến lược Epoch 8: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:37<00:00,  1.46it/s, E=0.45, H=0.05, loss=0.3310]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6443] | [Hate Acc: 0.7791]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7117

--- Chiến lược Epoch 9: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=0.11, H=0.28, loss=0.1623]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6385] | [Hate Acc: 0.7737]

--- Chiến lược Epoch 10: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=0.18, H=0.10, loss=0.1520]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6370] | [Hate Acc: 0.7764]

# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4257.72it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.42      0.42      0.42        40
     Disgust       0.57      0.53      0.55       132
   Enjoyment       0.71      0.70      0.70       193
        Fear       0.62      0.74      0.67        46
       Other       0.58      0.52      0.55       129
     Sadness       0.68      0.69      0.69       116
    Surprise       0.54      0.76      0.63        37

    accuracy                           0.62       693
   macro avg       0.59      0.62      0.60       693
weighted avg       0.62      0.62      0.62       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.85      0.86      0.85       600
   Offensive       0.74      0.71      0.72       496
        Hate       0.77      0.80      0.78       380

    accuracy                           0.79      1476
   macro avg       0.79      0.79      0.79      1476
weighted avg       0.79      0.79      0.79      1476
