## Module 1: 
# Epoch 1-3:  E:0.5 - H:0.5 
# Epoch 4-10:  E:0.6 - H:0.4 

# ----------------Train---------------------------------------#
# Best model: Epoch 5 with Avg Acc: 0.7106
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8024.10it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |


--- Chiến lược Epoch 1: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=0.81, H=0.67, loss=0.7387]
📊 Kết quả Epoch 1: [Emotion Acc: 0.5831] | [Hate Acc: 0.7005]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6418

--- Chiến lược Epoch 2: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=1.38, H=0.18, loss=0.7784]
📊 Kết quả Epoch 2: [Emotion Acc: 0.6166] | [Hate Acc: 0.7304]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6735

--- Chiến lược Epoch 3: Cân bằng (Warm-up) [E:0.5 - H:0.5] ---
✨ Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.05, H=1.24, loss=0.6473]
📊 Kết quả Epoch 3: [Emotion Acc: 0.6079] | [Hate Acc: 0.7575]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6827

--- Chiến lược Epoch 4: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.97, H=0.21, loss=0.6661]
📊 Kết quả Epoch 4: [Emotion Acc: 0.6239] | [Hate Acc: 0.7575]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6907

--- Chiến lược Epoch 5: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.19, H=0.03, loss=0.1279]
📊 Kết quả Epoch 5: [Emotion Acc: 0.6327] | [Hate Acc: 0.7886]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7106

--- Chiến lược Epoch 6: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.40it/s, E=0.34, H=0.17, loss=0.2723]
📊 Kết quả Epoch 6: [Emotion Acc: 0.6224] | [Hate Acc: 0.7791]

--- Chiến lược Epoch 7: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.82, H=0.05, loss=0.5131]
📊 Kết quả Epoch 7: [Emotion Acc: 0.6108] | [Hate Acc: 0.7818]

--- Chiến lược Epoch 8: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.40it/s, E=0.02, H=0.04, loss=0.0299]
📊 Kết quả Epoch 8: [Emotion Acc: 0.6356] | [Hate Acc: 0.7805]

--- Chiến lược Epoch 9: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.12, H=0.02, loss=0.0781]
📊 Kết quả Epoch 9: [Emotion Acc: 0.6108] | [Hate Acc: 0.7832]

--- Chiến lược Epoch 10: Tập trung Emotion (Fine-tuning) [E:0.6 - H:0.4] ---
✨ Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████| 670/670 [08:00<00:00,  1.39it/s, E=0.32, H=0.02, loss=0.1973]
📊 Kết quả Epoch 10: [Emotion Acc: 0.6224] | [Hate Acc: 0.7791]

# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3816.84it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.45      0.42      0.44        40
     Disgust       0.54      0.65      0.59       132
   Enjoyment       0.71      0.70      0.70       193
        Fear       0.61      0.67      0.64        46
       Other       0.62      0.50      0.56       129
     Sadness       0.71      0.67      0.69       116
    Surprise       0.62      0.68      0.65        37

    accuracy                           0.63       693
   macro avg       0.61      0.61      0.61       693
weighted avg       0.63      0.63      0.63       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.89      0.81      0.85       600
   Offensive       0.69      0.77      0.73       496
        Hate       0.80      0.79      0.80       380

    accuracy                           0.79      1476
   macro avg       0.80      0.79      0.79      1476
weighted avg       0.80      0.79      0.80      1476
