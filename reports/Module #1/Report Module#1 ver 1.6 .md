## Module 1: Chỉnh sửa HÀm mất mát
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-10:  E:0.7 - H:0.3 
# ----------------Train---------------------------------------#
# Best model: Epoch 6 with 0.7038
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4283.57it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |

--- Chiến lược Epoch 1: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [07:35<00:00,  1.47it/s, E=1.99, H=0.61, loss=1.3010]
📊 Kết quả Epoch 1: [Emotion Acc: 0.5408] | [Hate Acc: 0.6843]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6125

--- Chiến lược Epoch 2: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 2/10: 100%|████████████████████████████████████| 670/670 [07:44<00:00,  1.44it/s, E=0.80, H=1.23, loss=1.0129]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5729] | [Hate Acc: 0.7534]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6631

--- Chiến lược Epoch 3: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 3/10: 100%|████████████████████████████████████| 670/670 [07:51<00:00,  1.42it/s, E=0.92, H=0.66, loss=0.7912]
📊 Kết quả Epoch 3: [Emotion Acc: 0.6006] | [Hate Acc: 0.7656]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6831

--- Chiến lược Epoch 4: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 4/10: 100%|████████████████████████████████████| 670/670 [07:54<00:00,  1.41it/s, E=0.52, H=0.36, loss=0.4710]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5685] | [Hate Acc: 0.7683]

--- Chiến lược Epoch 5: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 5/10: 100%|████████████████████████████████████| 670/670 [07:57<00:00,  1.40it/s, E=0.14, H=0.13, loss=0.1353]
📊 Kết quả Epoch 5: [Emotion Acc: 0.5962] | [Hate Acc: 0.7615]

--- Chiến lược Epoch 6: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 6/10: 100%|████████████████████████████████████| 670/670 [07:55<00:00,  1.41it/s, E=0.16, H=0.14, loss=0.1536]
📊 Kết quả Epoch 6: [Emotion Acc: 0.6312] | [Hate Acc: 0.7764]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7038

--- Chiến lược Epoch 7: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 7/10: 100%|████████████████████████████████████| 670/670 [07:49<00:00,  1.43it/s, E=0.08, H=0.75, loss=0.2810]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6064] | [Hate Acc: 0.7778]

--- Chiến lược Epoch 8: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 8/10: 100%|████████████████████████████████████| 670/670 [07:45<00:00,  1.44it/s, E=0.05, H=0.30, loss=0.1213]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6122] | [Hate Acc: 0.7683]

--- Chiến lược Epoch 9: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 9/10: 100%|████████████████████████████████████| 670/670 [07:41<00:00,  1.45it/s, E=0.09, H=0.09, loss=0.0869]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6093] | [Hate Acc: 0.7751]

--- Chiến lược Epoch 10: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 10/10: 100%|███████████████████████████████████| 670/670 [07:46<00:00,  1.44it/s, E=0.03, H=0.15, loss=0.0684]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6093] | [Hate Acc: 0.7737]


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3171.31it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.48      0.50      0.49        40
     Disgust       0.64      0.58      0.61       132
   Enjoyment       0.72      0.70      0.71       193
        Fear       0.55      0.74      0.63        46
       Other       0.54      0.55      0.55       129
     Sadness       0.68      0.68      0.68       116
    Surprise       0.71      0.65      0.68        37

    accuracy                           0.63       693
   macro avg       0.62      0.63      0.62       693
weighted avg       0.64      0.63      0.64       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.84      0.88      0.86       600
   Offensive       0.74      0.73      0.74       496
        Hate       0.82      0.78      0.80       380

    accuracy                           0.80      1476
   macro avg       0.80      0.80      0.80      1476
weighted avg       0.80      0.80      0.80      1476

