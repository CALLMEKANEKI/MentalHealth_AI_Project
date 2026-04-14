## Module 1: Chỉnh sửa Batch Loss, Head Architecture, Differential Learning Rate, Label Smoothing, 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-12:  E:0.7 - H:0.3 
# Epoch 12-15:  E:0.8 - H:0.2 
# Note: Not update "Xử lý Teencode" and file _clean
# ----------------Train---------------------------------------#
# Best model: Epoch 15 with Avg Acc: 0.7106
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4791.34it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/15: 100%|████████████████████████████████████| 893/893 [07:57<00:00,  1.87it/s, E=1.85, H=1.09, loss=1.4694]
📊 Kết quả Epoch 1: [Emotion Acc: 0.4504] | [Hate Acc: 0.6233]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5369

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/15: 100%|████████████████████████████████████| 893/893 [07:58<00:00,  1.87it/s, E=1.26, H=1.47, loss=1.3651]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5190] | [Hate Acc: 0.6978]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6084

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/15: 100%|████████████████████████████████████| 893/893 [07:58<00:00,  1.87it/s, E=0.63, H=0.65, loss=0.6411]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5466] | [Hate Acc: 0.7168]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6317

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/15: 100%|████████████████████████████████████| 893/893 [08:00<00:00,  1.86it/s, E=1.28, H=0.78, loss=1.1297]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5875] | [Hate Acc: 0.7575]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6725

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/15: 100%|████████████████████████████████████| 893/893 [07:59<00:00,  1.86it/s, E=1.06, H=0.65, loss=0.9392]
📊 Kết quả Epoch 5: [Emotion Acc: 0.5962] | [Hate Acc: 0.7412]

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/15: 100%|████████████████████████████████████| 893/893 [07:58<00:00,  1.86it/s, E=1.26, H=0.42, loss=1.0101]
📊 Kết quả Epoch 6: [Emotion Acc: 0.5948] | [Hate Acc: 0.7629]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6788

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/15: 100%|████████████████████████████████████| 893/893 [08:02<00:00,  1.85it/s, E=0.94, H=0.49, loss=0.8043]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6210] | [Hate Acc: 0.7683]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6946

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/15: 100%|████████████████████████████████████| 893/893 [08:01<00:00,  1.85it/s, E=1.65, H=0.36, loss=1.2654]
📊 Kết quả Epoch 8: [Emotion Acc: 0.5933] | [Hate Acc: 0.7669]

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/15: 100%|████████████████████████████████████| 893/893 [08:01<00:00,  1.86it/s, E=0.71, H=0.37, loss=0.6097]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6166] | [Hate Acc: 0.7859]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7013

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/15: 100%|███████████████████████████████████| 893/893 [08:01<00:00,  1.86it/s, E=1.13, H=0.31, loss=0.8802]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6327] | [Hate Acc: 0.7805]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7066

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/15: 100%|███████████████████████████████████| 893/893 [07:58<00:00,  1.86it/s, E=0.91, H=0.32, loss=0.7327]
📊 Kết quả Epoch 11: [Emotion Acc: 0.6312] | [Hate Acc: 0.7805]

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/15: 100%|███████████████████████████████████| 893/893 [07:58<00:00,  1.87it/s, E=0.85, H=0.31, loss=0.6905]
📊 Kết quả Epoch 12: [Emotion Acc: 0.6239] | [Hate Acc: 0.7859]

--- Chiến lược Epoch 13: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 13/15: 100%|███████████████████████████████████| 893/893 [08:01<00:00,  1.86it/s, E=0.51, H=0.50, loss=0.5119]
📊 Kết quả Epoch 13: [Emotion Acc: 0.6341] | [Hate Acc: 0.7805]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7073

--- Chiến lược Epoch 14: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 14/15: 100%|███████████████████████████████████| 893/893 [08:02<00:00,  1.85it/s, E=0.40, H=0.37, loss=0.3965]
📊 Kết quả Epoch 14: [Emotion Acc: 0.6327] | [Hate Acc: 0.7818]

--- Chiến lược Epoch 15: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 15/15: 100%|███████████████████████████████████| 893/893 [08:02<00:00,  1.85it/s, E=0.90, H=0.36, loss=0.7920]
📊 Kết quả Epoch 15: [Emotion Acc: 0.6312] | [Hate Acc: 0.7900]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7106


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3913.07it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.44      0.45      0.44        40
     Disgust       0.58      0.58      0.58       132
   Enjoyment       0.75      0.70      0.72       193
        Fear       0.60      0.76      0.67        46
       Other      0 0.60      0.60      0.6       129
     Sadness       0.72      0.72      0.72       116
    Surprise       0.69      0.65      0.67        37

    accuracy                           0.65       693
   macro avg       0.62      0.64      0.63       693
weighted avg       0.65      0.65      0.65       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.85      0.88      0.87       600
   Offensive       0.76      0.74      0.75       496
        Hate       0.82      0.79      0.80       380

    accuracy                           0.81      1476
   macro avg       0.81      0.80      0.81      1476
weighted avg       0.81      0.81      0.81      1476
