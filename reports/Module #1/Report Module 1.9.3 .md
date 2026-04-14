## Module 1: Chỉnh sửa Batch Loss, Head Architecture, Differential Learning Rate, Label Smoothing, 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-12:  E:0.7 - H:0.3 
# Epoch 12-15:  E:0.8 - H:0.2 
# Note: Update "Xử lý Teencode" lần 2 and sửa lại file preprocess như ban đầu
# ----------------Train---------------------------------------#
# Best model: Epoch 12 with Avg Acc: 0.7146
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6094.41it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/15: 100%|████████████████████████████████████| 893/893 [07:56<00:00,  1.87it/s, E=2.19, H=1.16, loss=1.6764]
📊 Kết quả Epoch 1: [Emotion Acc: 0.3761] | [Hate Acc: 0.6260]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5011

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/15: 100%|████████████████████████████████████| 893/893 [08:02<00:00,  1.85it/s, E=1.46, H=1.03, loss=1.2459]
📊 Kết quả Epoch 2: [Emotion Acc: 0.5408] | [Hate Acc: 0.7019]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6214

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/15: 100%|████████████████████████████████████| 893/893 [07:58<00:00,  1.87it/s, E=1.36, H=0.51, loss=0.9366]
📊 Kết quả Epoch 3: [Emotion Acc: 0.5700] | [Hate Acc: 0.7249]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6475

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/15: 100%|████████████████████████████████████| 893/893 [07:58<00:00,  1.87it/s, E=1.55, H=0.55, loss=1.2541]
📊 Kết quả Epoch 4: [Emotion Acc: 0.5816] | [Hate Acc: 0.7425]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6621

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/15: 100%|████████████████████████████████████| 893/893 [07:57<00:00,  1.87it/s, E=1.66, H=0.75, loss=1.3866]
📊 Kết quả Epoch 5: [Emotion Acc: 0.5685] | [Hate Acc: 0.7385]

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/15: 100%|████████████████████████████████████| 893/893 [07:59<00:00,  1.86it/s, E=0.69, H=0.88, loss=0.7492]
📊 Kết quả Epoch 6: [Emotion Acc: 0.5554] | [Hate Acc: 0.7236]

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/15: 100%|████████████████████████████████████| 893/893 [08:00<00:00,  1.86it/s, E=0.95, H=0.35, loss=0.7701]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6079] | [Hate Acc: 0.7710]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6894

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/15: 100%|████████████████████████████████████| 893/893 [07:59<00:00,  1.86it/s, E=0.82, H=0.44, loss=0.7056]
📊 Kết quả Epoch 8: [Emotion Acc: 0.5904] | [Hate Acc: 0.7588]

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/15: 100%|████████████████████████████████████| 893/893 [08:03<00:00,  1.85it/s, E=0.49, H=0.41, loss=0.4670]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6122] | [Hate Acc: 0.7751]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6937

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/15: 100%|███████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.70, H=0.39, loss=0.6079]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6108] | [Hate Acc: 0.7859]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6983

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/15: 100%|███████████████████████████████████| 893/893 [08:09<00:00,  1.83it/s, E=0.84, H=0.31, loss=0.6793]
📊 Kết quả Epoch 11: [Emotion Acc: 0.6093] | [Hate Acc: 0.7859]

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/15: 100%|███████████████████████████████████| 893/893 [08:06<00:00,  1.83it/s, E=0.62, H=0.43, loss=0.5611]
📊 Kết quả Epoch 12: [Emotion Acc: 0.6181] | [Hate Acc: 0.7832]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7006

--- Chiến lược Epoch 13: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 13/15: 100%|███████████████████████████████████| 893/893 [08:03<00:00,  1.85it/s, E=0.88, H=0.48, loss=0.7989]
📊 Kết quả Epoch 13: [Emotion Acc: 0.6312] | [Hate Acc: 0.7981]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7146

--- Chiến lược Epoch 14: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 14/15: 100%|███████████████████████████████████| 893/893 [08:04<00:00,  1.84it/s, E=0.61, H=0.31, loss=0.5505]
📊 Kết quả Epoch 14: [Emotion Acc: 0.6297] | [Hate Acc: 0.7846]

--- Chiến lược Epoch 15: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 15/15: 100%|███████████████████████████████████| 893/893 [08:03<00:00,  1.85it/s, E=0.94, H=0.49, loss=0.8461]
📊 Kết quả Epoch 15: [Emotion Acc: 0.6283] | [Hate Acc: 0.7886]



# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3458.34it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.41      0.45      0.43        40
     Disgust       0.59      0.57      0.58       132
   Enjoyment       0.70      0.75      0.73       193
        Fear       0.67      0.80      0.73        46
       Other       0.62      0.50      0.55       129
     Sadness       0.67      0.67      0.67       116
    Surprise       0.64      0.73      0.68        37

    accuracy                           0.64       693
   macro avg       0.62      0.64      0.62       693
weighted avg       0.64      0.64      0.64       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.84      0.88      0.86       600
   Offensive       0.72      0.71      0.72       496
        Hate       0.80      0.77      0.78       380

    accuracy                           0.79      1476
   macro avg       0.79      0.78      0.79      1476
weighted avg       0.79      0.79      0.79      1476

