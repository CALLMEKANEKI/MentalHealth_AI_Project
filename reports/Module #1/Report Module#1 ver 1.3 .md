## Module 1: E:0.7 - H:0.3 N=1

# ----------------Train---------------------------------------#
# Best model: Epoch 8(0.7055)  Epoch 10(0.7102)
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6292.53it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |

✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=1.97, H=0.70, loss=1.5909]
📊 Accuracy - [Emotion: 0.5831] | [Hate: 0.6748]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6289
✨ Epoch 2/10: 100%|████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=0.08, H=0.95, loss=0.3429]
📊 Accuracy - [Emotion: 0.5860] | [Hate: 0.7168]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6514
✨ Epoch 3/10: 100%|████████████████████████████████████| 670/670 [07:36<00:00,  1.47it/s, E=0.32, H=0.71, loss=0.4376]
📊 Accuracy - [Emotion: 0.6327] | [Hate: 0.7344]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6835
✨ Epoch 4/10: 100%|████████████████████████████████████| 670/670 [07:45<00:00,  1.44it/s, E=0.23, H=0.28, loss=0.2473]
📊 Accuracy - [Emotion: 0.6356] | [Hate: 0.7344]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6850
✨ Epoch 5/10: 100%|████████████████████████████████████| 670/670 [07:41<00:00,  1.45it/s, E=0.03, H=0.23, loss=0.0916]
📊 Accuracy - [Emotion: 0.6283] | [Hate: 0.7425]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6854
✨ Epoch 6/10: 100%|████████████████████████████████████| 670/670 [07:49<00:00,  1.43it/s, E=0.12, H=0.74, loss=0.3045]
📊 Accuracy - [Emotion: 0.6341] | [Hate: 0.7629]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6985
✨ Epoch 7/10: 100%|████████████████████████████████████| 670/670 [07:48<00:00,  1.43it/s, E=0.02, H=0.36, loss=0.1248]
📊 Accuracy - [Emotion: 0.6166] | [Hate: 0.7737]
✨ Epoch 8/10: 100%|████████████████████████████████████| 670/670 [07:47<00:00,  1.43it/s, E=0.11, H=0.04, loss=0.0905]
📊 Accuracy - [Emotion: 0.6399] | [Hate: 0.7710]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7055
✨ Epoch 9/10: 100%|████████████████████████████████████| 670/670 [07:49<00:00,  1.43it/s, E=0.04, H=0.77, loss=0.2603]
📊 Accuracy - [Emotion: 0.6312] | [Hate: 0.7710]
✨ Epoch 10/10: 100%|███████████████████████████████████| 670/670 [07:54<00:00,  1.41it/s, E=0.02, H=0.23, loss=0.0833]
📊 Accuracy - [Emotion: 0.6399] | [Hate: 0.7805]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7102


# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3923.23it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.42      0.38      0.39        40
     Disgust       0.57      0.59      0.58       132
   Enjoyment       0.71      0.71      0.71       193
        Fear       0.64      0.76      0.69        46
       Other       0.58      0.50      0.53       129
     Sadness       0.67      0.71      0.69       116
    Surprise       0.62      0.65      0.63        37

    accuracy                           0.63       693
   macro avg       0.60      0.61      0.60       693
weighted avg       0.63      0.63      0.63       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.86      0.85      0.85       600
   Offensive       0.72      0.77      0.74       496
        Hate       0.83      0.78      0.80       380

    accuracy                           0.80      1476
   macro avg       0.80      0.80      0.80      1476
weighted avg       0.81      0.80      0.80      1476

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .