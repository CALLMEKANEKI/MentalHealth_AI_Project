## Module 1: Chỉnh sửa Batch Loss, Head Architecture, Differential Learning Rate, Label Smoothing, 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-10:  E:0.7 - H:0.3 
# ----------------Train---------------------------------------#
# Best model: Epoch 9 with Avg Acc: 0.6981
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 5908.47it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.bias                    | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

--- Chiến lược Epoch 1: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [07:37<00:00,  1.46it/s, E=1.70, H=0.86, loss=1.2793]
📊 Kết quả Epoch 1: [Emotion Acc: 0.4300] | [Hate Acc: 0.6463]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5382

--- Chiến lược Epoch 2: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 2/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=1.25, H=0.39, loss=0.8196]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5641] | [Hate Acc: 0.7114]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6378

--- Chiến lược Epoch 3: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 3/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=1.64, H=0.54, loss=1.0866]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5685] | [Hate Acc: 0.7304]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6494

--- Chiến lược Epoch 4: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 4/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=0.78, H=0.44, loss=0.6786]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5685] | [Hate Acc: 0.7439]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6562

--- Chiến lược Epoch 5: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 5/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=1.22, H=0.56, loss=1.0213]
📊 Kết quả Epoch 5: [Emotion Acc: 0.5743] | [Hate Acc: 0.7439]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6591

--- Chiến lược Epoch 6: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 6/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=0.66, H=0.71, loss=0.6781]
📊 Kết quả Epoch 6: [Emotion Acc: 0.5933] | [Hate Acc: 0.7710]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6821

--- Chiến lược Epoch 7: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 7/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=0.66, H=0.42, loss=0.5883]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6181] | [Hate Acc: 0.7683]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6932

--- Chiến lược Epoch 8: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 8/10: 100%|████████████████████████████████████| 670/670 [07:38<00:00,  1.46it/s, E=0.47, H=0.35, loss=0.4310]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6093] | [Hate Acc: 0.7534]

--- Chiến lược Epoch 9: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 9/10: 100%|████████████████████████████████████| 670/670 [07:39<00:00,  1.46it/s, E=0.77, H=0.31, loss=0.6309]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6224] | [Hate Acc: 0.7737]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6981

--- Chiến lược Epoch 10: Tập trung Emotion (Fine-tuning) [E:0.7 - H:0.3] ---
✨ Epoch 10/10: 100%|███████████████████████████████████| 670/670 [07:39<00:00,  1.46it/s, E=0.41, H=0.38, loss=0.4045]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6210] | [Hate Acc: 0.7737]


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6488.19it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.39      0.53      0.45        40
     Disgust       0.57      0.51      0.54       132
   Enjoyment       0.74      0.66      0.70       193
        Fear       0.58      0.74      0.65        46
       Other       0.58      0.62      0.60       129
     Sadness       0.71      0.67      0.69       116
    Surprise       0.60      0.70      0.65        37

    accuracy                           0.63       693
   macro avg       0.60      0.63      0.61       693
weighted avg       0.64      0.63      0.63       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.85      0.86      0.85       600
   Offensive       0.74      0.71      0.72       496
        Hate       0.80      0.83      0.81       380

    accuracy                           0.80      1476
   macro avg       0.80      0.80      0.80      1476
weighted avg       0.80      0.80      0.80      1476
