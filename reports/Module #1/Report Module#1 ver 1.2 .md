## Module 1: E:0.6 - H:0.4 N=1

# ----------------Train---------------------------------------#
# Best model: Epoch 5(0.7091)  Epoch 8(0.7116)
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 7854.43it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.bias                    | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

✨ Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:59<00:00,  1.40it/s, E=1.14, H=0.80, loss=1.0017]
📊 Accuracy - [Emotion: 0.6050] | [Hate: 0.7019]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6534
✨ Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:01<00:00,  1.39it/s, E=0.55, H=0.75, loss=0.6341]
📊 Accuracy - [Emotion: 0.5918] | [Hate: 0.7127]
✨ Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=0.50, H=0.63, loss=0.5538]
📊 Accuracy - [Emotion: 0.6166] | [Hate: 0.7561]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6864
✨ Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:01<00:00,  1.39it/s, E=0.05, H=0.18, loss=0.0993]
📊 Accuracy - [Emotion: 0.6166] | [Hate: 0.7710]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6938
✨ Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:01<00:00,  1.39it/s, E=0.37, H=0.18, loss=0.2927]
📊 Accuracy - [Emotion: 0.6283] | [Hate: 0.7900]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7091
✨ Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:01<00:00,  1.39it/s, E=0.08, H=0.09, loss=0.0864]
📊 Accuracy - [Emotion: 0.6312] | [Hate: 0.7737]
✨ Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=0.42, H=0.07, loss=0.2803]
📊 Accuracy - [Emotion: 0.6254] | [Hate: 0.7818]
✨ Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.39it/s, E=0.04, H=0.26, loss=0.1286]
📊 Accuracy - [Emotion: 0.6399] | [Hate: 0.7832]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7116
✨ Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.39it/s, E=0.09, H=0.03, loss=0.0674]
📊 Accuracy - [Emotion: 0.6327] | [Hate: 0.7805]
✨ Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=0.01, H=0.02, loss=0.0161]
📊 Accuracy - [Emotion: 0.6297] | [Hate: 0.7832]


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 9736.22it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.39      0.40      0.40        40
     Disgust       0.61      0.60      0.60       132
   Enjoyment       0.72      0.72      0.72       193
        Fear       0.61      0.76      0.68        46
       Other       0.56      0.54      0.55       129
     Sadness       0.71      0.72      0.71       116
    Surprise       0.66      0.57      0.61        37

    accuracy                           0.64       693
   macro avg       0.61      0.61      0.61       693
weighted avg       0.64      0.64      0.64       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.88      0.85      0.86       600
   Offensive       0.73      0.77      0.75       496
        Hate       0.79      0.78      0.79       380

    accuracy                           0.81      1476
   macro avg       0.80      0.80      0.80      1476
weighted avg       0.81      0.81      0.81      1476

