## Module 1: Chỉnh sửa Batch Loss, Head Architecture, Differential Learning Rate, Label Smoothing, 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-12:  E:0.7 - H:0.3 
# Epoch 12-15:  E:0.8 - H:0.2 
# Note: Update "Xử lý Teencode" and file _clean
# ----------------Train---------------------------------------#
# Best model: Epoch 12 with Avg Acc: 0.7063
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 5472.18it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |


--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/15: 100%|████████████████████████████████████| 893/893 [08:03<00:00,  1.85it/s, E=2.00, H=0.69, loss=1.3438]
📊 Kết quả Epoch 1: [Emotion Acc: 0.4490] | [Hate Acc: 0.6233]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5361

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/15: 100%|████████████████████████████████████| 893/893 [08:07<00:00,  1.83it/s, E=1.62, H=0.51, loss=1.0662]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5321] | [Hate Acc: 0.6572]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5946

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/15: 100%|████████████████████████████████████| 893/893 [08:08<00:00,  1.83it/s, E=1.23, H=0.53, loss=0.8788]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5685] | [Hate Acc: 0.7019]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6352

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/15: 100%|████████████████████████████████████| 893/893 [08:13<00:00,  1.81it/s, E=0.77, H=0.55, loss=0.7066]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5685] | [Hate Acc: 0.7439]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6562

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/15: 100%|████████████████████████████████████| 893/893 [08:34<00:00,  1.74it/s, E=1.03, H=0.62, loss=0.9046]
📊 Kết quả Epoch 5: [Emotion Acc: 0.5933] | [Hate Acc: 0.7588]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6761

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/15: 100%|████████████████████████████████████| 893/893 [08:32<00:00,  1.74it/s, E=1.03, H=0.32, loss=0.8203]
📊 Kết quả Epoch 6: [Emotion Acc: 0.6035] | [Hate Acc: 0.7615]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6825

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/15: 100%|████████████████████████████████████| 893/893 [08:28<00:00,  1.75it/s, E=0.94, H=0.33, loss=0.7586]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6137] | [Hate Acc: 0.7602]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6869

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/15: 100%|████████████████████████████████████| 893/893 [08:25<00:00,  1.77it/s, E=1.03, H=0.31, loss=0.8136]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6210] | [Hate Acc: 0.7913]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7062

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/15: 100%|████████████████████████████████████| 893/893 [08:11<00:00,  1.82it/s, E=0.45, H=0.32, loss=0.4093]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6137] | [Hate Acc: 0.7832]

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/15: 100%|███████████████████████████████████| 893/893 [08:16<00:00,  1.80it/s, E=1.10, H=0.80, loss=1.0102]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6166] | [Hate Acc: 0.7900]

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/15: 100%|███████████████████████████████████| 893/893 [08:19<00:00,  1.79it/s, E=0.88, H=0.43, loss=0.7415]
📊 Kết quả Epoch 11: [Emotion Acc: 0.6108] | [Hate Acc: 0.7954]

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/15: 100%|███████████████████████████████████| 893/893 [08:22<00:00,  1.78it/s, E=0.53, H=0.35, loss=0.4770]
📊 Kết quả Epoch 12: [Emotion Acc: 0.6254] | [Hate Acc: 0.7873]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7063

--- Chiến lược Epoch 13: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 13/15: 100%|███████████████████████████████████| 893/893 [08:21<00:00,  1.78it/s, E=0.88, H=0.39, loss=0.7798]
📊 Kết quả Epoch 13: [Emotion Acc: 0.6166] | [Hate Acc: 0.7913]

--- Chiến lược Epoch 14: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 14/15: 100%|███████████████████████████████████| 893/893 [08:17<00:00,  1.79it/s, E=0.72, H=0.32, loss=0.6378]
📊 Kết quả Epoch 14: [Emotion Acc: 0.6195] | [Hate Acc: 0.7846]

--- Chiến lược Epoch 15: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 15/15: 100%|███████████████████████████████████| 893/893 [08:25<00:00,  1.77it/s, E=0.58, H=0.30, loss=0.5254]
📊 Kết quả Epoch 15: [Emotion Acc: 0.6108] | [Hate Acc: 0.7886]


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6715.53it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.51      0.45      0.48        40
     Disgust       0.55      0.62      0.59       132
   Enjoyment       0.70      0.66      0.68       193
        Fear       0.66      0.76      0.71        46
       Other       0.58      0.49      0.53       129
     Sadness       0.62      0.70      0.66       116
    Surprise       0.54      0.54      0.54        37

    accuracy                           0.62       693
   macro avg       0.60      0.60      0.60       693
weighted avg       0.62      0.62      0.61       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.87      0.85      0.86       600
   Offensive       0.73      0.75      0.74       496
        Hate       0.81      0.81      0.81       380

    accuracy                           0.81      1476
   macro avg       0.80      0.80      0.80      1476
weighted avg       0.81      0.81      0.81      1476















