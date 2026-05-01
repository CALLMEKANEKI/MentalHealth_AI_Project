
## Trainning
📊 Emotion classes: {'Anger': 0, 'Disgust': 1, 'Enjoyment': 2, 'Fear': 3, 'Other': 4, 'Sadness': 5, 'Surprise': 6}
⚖️ Emotion loss weights: [2.027037   0.74002934 0.5087108  2.492363   0.77626973 0.83692867
 3.2750885 ]
✅ WeightedRandomSampler được khởi tạo với 5548 mẫu emotion, 5163 mẫu hate
⏳ Tokenizing dataset... (chỉ một lần)
✅ Tokenization hoàn tất!
⏳ Tokenizing dataset... (chỉ một lần)
✅ Tokenization hoàn tất!
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4543.89it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

✨ Epoch 1/20: 100%|████████████████████████████████████| 893/893 [08:04<00:00,  1.84it/s, E=1.02, H=0.00, loss=1.0214]

📊 Epoch 1: [Emotion Acc: 0.4038] | [Hate Acc: 0.5935] | Avg: 0.4986
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.4038)
⭐ Lưu best avg model (avg=0.4986)
✨ Epoch 2/20: 100%|████████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.79, H=1.20, loss=1.9847]

📊 Epoch 2: [Emotion Acc: 0.5292] | [Hate Acc: 0.6789] | Avg: 0.6040
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.5292)
⭐ Lưu best avg model (avg=0.6040)
✨ Epoch 3/20: 100%|████████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.46, H=0.00, loss=0.4623]

📊 Epoch 3: [Emotion Acc: 0.5248] | [Hate Acc: 0.6951] | Avg: 0.6100
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best avg model (avg=0.6100)
✨ Epoch 4/20: 100%|████████████████████████████████████| 893/893 [08:07<00:00,  1.83it/s, E=0.31, H=0.00, loss=0.3119]

📊 Epoch 4: [Emotion Acc: 0.5773] | [Hate Acc: 0.6951] | Avg: 0.6362
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.5773)
⭐ Lưu best avg model (avg=0.6362)
✨ Epoch 5/20: 100%|████████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.28, H=0.18, loss=0.4624]

📊 Epoch 5: [Emotion Acc: 0.5933] | [Hate Acc: 0.7263] | Avg: 0.6598
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.5933)
⭐ Lưu best avg model (avg=0.6598)
✨ Epoch 6/20: 100%|████████████████████████████████████| 893/893 [08:05<00:00,  1.84it/s, E=0.33, H=0.47, loss=0.8065]

📊 Epoch 6: [Emotion Acc: 0.5904] | [Hate Acc: 0.7358] | Avg: 0.6631
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best avg model (avg=0.6631)
✨ Epoch 7/20: 100%|████████████████████████████████████| 893/893 [08:05<00:00,  1.84it/s, E=0.37, H=0.00, loss=0.3679]

📊 Epoch 7: [Emotion Acc: 0.5962] | [Hate Acc: 0.7453] | Avg: 0.6707
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.5962)
⭐ Lưu best avg model (avg=0.6707)
✨ Epoch 8/20: 100%|████████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.27, H=0.00, loss=0.2664]

📊 Epoch 8: [Emotion Acc: 0.6137] | [Hate Acc: 0.7439] | Avg: 0.6788
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.6137)
⭐ Lưu best avg model (avg=0.6788)
✨ Epoch 9/20: 100%|████████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.28, H=0.18, loss=0.4584]

📊 Epoch 9: [Emotion Acc: 0.6166] | [Hate Acc: 0.7425] | Avg: 0.6796
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best emotion model (acc=0.6166)
⭐ Lưu best avg model (avg=0.6796)
✨ Epoch 10/20: 100%|███████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.23, H=0.17, loss=0.4013]

📊 Epoch 10: [Emotion Acc: 0.6152] | [Hate Acc: 0.7439] | Avg: 0.6795
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
✨ Epoch 11/20: 100%|███████████████████████████████████| 893/893 [08:06<00:00,  1.84it/s, E=0.30, H=0.00, loss=0.3039]

📊 Epoch 11: [Emotion Acc: 0.6050] | [Hate Acc: 0.7696] | Avg: 0.6873
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 2.00e-05
⭐ Lưu best avg model (avg=0.6873)
✨ Epoch 12/20: 100%|███████████████████████████████████| 893/893 [08:43<00:00,  1.71it/s, E=0.37, H=0.00, loss=0.3673]

📊 Epoch 12: [Emotion Acc: 0.6064] | [Hate Acc: 0.7710] | Avg: 0.6887
   Uncertainty weights - Emotion: 1.000, Hate: 1.000
   Current LR: 1.00e-05
⭐ Lưu best avg model (avg=0.6887)

🛑 Early stopping tại epoch 12 do emotion acc không cải thiện sau 3 epochs

✅ Hoàn thành training. Best emotion acc: 0.6166, Best avg acc: 0.6887
📁 Checkpoints lưu tại:
   - C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\checkpoints\best_multitask_model.pth
   - C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\checkpoints\best_emotion_model.pth


## Test Best Model
⏳ Tokenizing dataset... (chỉ một lần)
✅ Tokenization hoàn tất!
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3926.46it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |

📊 Đang dự đoán...
==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.56      0.38      0.45        40
     Disgust       0.57      0.61      0.59       132
   Enjoyment       0.68      0.72      0.70       193
        Fear       0.61      0.85      0.71        46
       Other       0.57      0.53      0.55       129
     Sadness       0.67      0.64      0.65       116
    Surprise       0.68      0.46      0.55        37

    accuracy                           0.62       693
   macro avg       0.62      0.60      0.60       693
weighted avg       0.62      0.62      0.62       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.89      0.82      0.85       599
   Offensive       0.72      0.65      0.69       496
        Hate       0.70      0.87      0.78       380

    accuracy                           0.78      1475
   macro avg       0.77      0.78      0.77      1475
weighted avg       0.78      0.78      0.78      1475

## Test Best Emotion Model
⏳ Tokenizing dataset... (chỉ một lần)
✅ Tokenization hoàn tất!
Loading weights: 100%|████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 10592.35it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |

📊 Đang dự đoán...

==================== EMOTION REPORT ====================
              precision    recall  f1-score   support

       Anger       0.50      0.53      0.51        40
     Disgust       0.52      0.64      0.58       132
   Enjoyment       0.73      0.65      0.68       193
        Fear       0.68      0.83      0.75        46
       Other       0.50      0.57      0.53       129
     Sadness       0.70      0.55      0.62       116
    Surprise       0.70      0.38      0.49        37

    accuracy                           0.61       693
   macro avg       0.62      0.59      0.59       693
weighted avg       0.62      0.61      0.61       693


==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.87      0.82      0.84       599
   Offensive       0.68      0.65      0.66       496
        Hate       0.71      0.82      0.76       380

    accuracy                           0.76      1475
   macro avg       0.75      0.76      0.76      1475
weighted avg       0.76      0.76      0.76      1475


