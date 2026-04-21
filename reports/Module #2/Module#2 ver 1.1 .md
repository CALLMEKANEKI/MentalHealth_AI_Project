## Module 2 ver 1.1: MultiLabelBinarizer, Pos_weight, BCEWithLogitsLoss, Top k accuracy, Differential Learning Rate, Gradient Clipping, Linear Warmup Scheduler

## ---Data---##
# Total_label: 58
# Min_sample: 50

## ---Config---##
# Total epoch: 20
# EPOCHS = 20
# BATCH_SIZE = 32
# MAX_LEN = 128
# LR_BACKBONE = 2e-5
# LR_HEAD = 5e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EARLY_STOP_PATIENCE = 5
# GRAD_CLIP = 1.0
# WEIGHT_DECAY = 0.01

# Best model: Epoch 14 with Val F1 micro=0.7615, macro=0.7669. Top-1: 0.7967, Top-5: 0.9562, Top-10: 0.9754

# ----------------Train---------------------------------------#
Số lượng nhãn: 58
Ví dụ 10 nhãn đầu: ['Bệnh Alzheimer', 'Bệnh CreutzfeldtJakob', 'Bệnh Parkinson', 'Bệnh rối loạn tâm thần ở tuổi già', 'Bệnh tâm thần', 'Chán ăn tâm thần', 'Chứng háu ăn tâm thần', 'Chứng nghiện', 'Chứng sợ không gian hẹp (Claustrophobia)', 'Căng thẳng Stress']
pos_weight (top 10): tensor([193.7714, 116.5345, 102.2879, 173.7949, 108.9516,  55.3388, 122.9454,
         36.2514,  55.3388,  73.0978], device='cuda:0')

Epoch 1/20: 100%|█████████████████████████████████████████████████████████| 214/214 [19:50<00:00,  5.56s/it, loss=1.44]
📊 Epoch 1: Loss=1.3448, Val F1 micro=0.1038, macro=0.1264
   Top-1: 0.1608, Top-5: 0.5086, Top-10: 0.7084
⭐ Saved best model (F1 micro=0.1038)
Epoch 2/20: 100%|█████████████████████████████████████████████████████████| 214/214 [19:51<00:00,  5.57s/it, loss=0.77]
📊 Epoch 2: Loss=0.8891, Val F1 micro=0.1868, macro=0.1900
   Top-1: 0.4743, Top-5: 0.7714, Top-10: 0.8741
⭐ Saved best model (F1 micro=0.1868)
Epoch 3/20: 100%|████████████████████████████████████████████████████████| 214/214 [18:57<00:00,  5.32s/it, loss=0.136]
📊 Epoch 3: Loss=0.4472, Val F1 micro=0.2973, macro=0.2777
   Top-1: 0.5729, Top-5: 0.8624, Top-10: 0.9398
⭐ Saved best model (F1 micro=0.2973)
Epoch 4/20: 100%|██████████████████████████████████████████████████████████| 214/214 [19:28<00:00,  5.46s/it, loss=1.8]
📊 Epoch 4: Loss=0.2922, Val F1 micro=0.3525, macro=0.3355
   Top-1: 0.6208, Top-5: 0.8891, Top-10: 0.9514
⭐ Saved best model (F1 micro=0.3525)
Epoch 5/20: 100%|███████████████████████████████████████████████████████| 214/214 [33:20<00:00,  9.35s/it, loss=0.0681]
📊 Epoch 5: Loss=0.2016, Val F1 micro=0.4677, macro=0.4812
   Top-1: 0.6995, Top-5: 0.9240, Top-10: 0.9651
⭐ Saved best model (F1 micro=0.4677)
Epoch 6/20: 100%|████████████████████████████████████████████████████████| 214/214 [39:22<00:00, 11.04s/it, loss=0.158]
📊 Epoch 6: Loss=0.1420, Val F1 micro=0.5579, macro=0.5711
   Top-1: 0.7201, Top-5: 0.9377, Top-10: 0.9671
⭐ Saved best model (F1 micro=0.5579)
Epoch 7/20: 100%|███████████████████████████████████████████████████████| 214/214 [20:30<00:00,  5.75s/it, loss=0.0278]
📊 Epoch 7: Loss=0.1109, Val F1 micro=0.5743, macro=0.5884
   Top-1: 0.7194, Top-5: 0.9309, Top-10: 0.9713
⭐ Saved best model (F1 micro=0.5743)
Epoch 8/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.00939]
📊 Epoch 8: Loss=0.0842, Val F1 micro=0.5920, macro=0.6120
   Top-1: 0.7303, Top-5: 0.9343, Top-10: 0.9685
⭐ Saved best model (F1 micro=0.5920)
Epoch 9/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.00224]
📊 Epoch 9: Loss=0.0679, Val F1 micro=0.6583, macro=0.6875
   Top-1: 0.7529, Top-5: 0.9514, Top-10: 0.9719
⭐ Saved best model (F1 micro=0.6583)
Epoch 10/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.00157]
📊 Epoch 10: Loss=0.0546, Val F1 micro=0.6954, macro=0.7061
   Top-1: 0.7666, Top-5: 0.9507, Top-10: 0.9754
⭐ Saved best model (F1 micro=0.6954)
Epoch 11/20: 100%|████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.000727]
📊 Epoch 11: Loss=0.0451, Val F1 micro=0.7234, macro=0.7323
   Top-1: 0.7762, Top-5: 0.9541, Top-10: 0.9726
⭐ Saved best model (F1 micro=0.7234)
Epoch 12/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.00119]
📊 Epoch 12: Loss=0.0390, Val F1 micro=0.7201, macro=0.7360
   Top-1: 0.7803, Top-5: 0.9528, Top-10: 0.9747
Epoch 13/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.00166]
📊 Epoch 13: Loss=0.0290, Val F1 micro=0.7449, macro=0.7501
   Top-1: 0.7851, Top-5: 0.9562, Top-10: 0.9740
⭐ Saved best model (F1 micro=0.7449)
Epoch 14/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:25<00:00,  5.45s/it, loss=0.0249]
📊 Epoch 14: Loss=0.0242, Val F1 micro=0.7615, macro=0.7669
   Top-1: 0.7967, Top-5: 0.9562, Top-10: 0.9754
⭐ Saved best model (F1 micro=0.7615)
Epoch 15/20:  18%|█████████▊            

# ----------------Test---------------------------------------#
Loaded 58 labels
Test samples: 1461
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3690.50it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base

# -------Result with test file------- #
✅ Test F1 micro: 0.7856, macro: 0.7771
   Top-1 accuracy: 0.8084
   Top-5 accuracy: 0.9439
   Top-10 accuracy: 0.9692
   Top-20 accuracy: 0.9877

# -------Result with random exsample-------#
📝 Dự đoán mẫu (top 5):

Text: Tôi cảm thấy buồn bã, mất ngủ, không muốn gặp ai, nghĩ đến cái chết....
Top 5: [('Rối loạn trầm cảm', 0.9795087575912476), ('Khác', 0.23137308657169342), ('Rối loạn giấc ngủ (Hypersomnia)', 0.053099967539310455), ('Rối loạn cảm xúc', 0.004647211637347937), ('Rối loạn lưỡng cực (Bipolar disorder)', 0.00378934177570045)]

Text: Tôi sợ đi thang máy, mỗi lần thấy nhện là hoảng hốt, tay run....
Top 5: [('Chứng sợ không gian hẹp (Claustrophobia)', 0.8970999121665955), ('Khác', 0.6784501075744629), ('Rối loạn hoảng sợ (Panic disorder)', 0.0008053441997617483), ('Rối loạn lo âu', 0.00048330536810681224), ('Hội chứng Fregoli', 0.0004760149458888918)]
