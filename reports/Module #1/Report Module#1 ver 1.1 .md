## Module 1: E:0.4 - H:0.6 N=1 ##

# ----------------Train---------------------------------------#
# Best model: Epoch 5(0.7075)  Epoch 9(0.7088)
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6387.44it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

✨ Epoch 1/10: 100%|████████████████████████████████████| 670/670 [08:02<00:00,  1.39it/s, E=1.14, H=0.60, loss=0.8183]
📊 Accuracy - [Emotion: 0.5350] | [Hate: 0.6572]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.5961
✨ Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:06<00:00,  1.38it/s, E=0.87, H=0.37, loss=0.5656]
📊 Accuracy - [Emotion: 0.5948] | [Hate: 0.7236]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6592
✨ Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:04<00:00,  1.38it/s, E=0.27, H=0.84, loss=0.6105]
📊 Accuracy - [Emotion: 0.6108] | [Hate: 0.7534]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6821
✨ Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.38it/s, E=1.27, H=0.02, loss=0.5200]
📊 Accuracy - [Emotion: 0.6283] | [Hate: 0.7669]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.6976
✨ Epoch 5/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.38it/s, E=0.43, H=0.55, loss=0.4977]
📊 Accuracy - [Emotion: 0.6385] | [Hate: 0.7764]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7075
✨ Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.39it/s, E=0.11, H=0.16, loss=0.1386]
📊 Accuracy - [Emotion: 0.6137] | [Hate: 0.7818]
✨ Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [08:03<00:00,  1.39it/s, E=0.15, H=0.02, loss=0.0713]
📊 Accuracy - [Emotion: 0.6195] | [Hate: 0.7602]
✨ Epoch 8/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:44<00:00,  1.44it/s, E=0.46, H=0.10, loss=0.2434]
📊 Accuracy - [Emotion: 0.6370] | [Hate: 0.7724]
✨ Epoch 9/10: 100%|████████████████████████████████████████████████████████████████████████| 670/670 [07:34<00:00,  1.47it/s, E=0.46, H=0.03, loss=0.2024]
📊 Accuracy - [Emotion: 0.6385] | [Hate: 0.7791]
⭐ Đã lưu Model xuất sắc nhất với Avg Acc: 0.7088
✨ Epoch 10/10: 100%|███████████████████████████████████████████████████████████████████████| 670/670 [07:33<00:00,  1.48it/s, E=0.04, H=0.87, loss=0.5374]
📊 Accuracy - [Emotion: 0.6356] | [Hate: 0.7737]



# ----------------Test---------------------------------------#
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8895.04it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.39      0.38      0.38        40
     Disgust       0.56      0.59      0.57       132
   Enjoyment       0.72      0.69      0.70       193
        Fear       0.68      0.70      0.69        46
       Other       0.53      0.57      0.55       129
     Sadness       0.71      0.69      0.70       116
    Surprise       0.69      0.59      0.64        37

    accuracy                           0.63       693
   macro avg       0.61      0.60      0.61       693
weighted avg       0.63      0.63      0.63       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.85      0.88      0.86       600
   Offensive       0.74      0.75      0.74       496
        Hate       0.84      0.77      0.80       380

    accuracy                           0.81      1476
   macro avg       0.81      0.80      0.80      1476
weighted avg       0.81      0.81      0.81      1476

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .

