## Module 2 ver 1.2: MultiLabelBinarizer, Pos_weight, BCEWithLogitsLoss, Top k accuracy, Differential Learning Rate, Gradient Clipping, Linear Warmup Scheduler

## ---Data---##
# Total_label: 58
# Min_sample: 30

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

# Best model: Epoch 14 with Val F1 micro=0.8024, macro=0.8106. Top-1: 0.8234, Top-5: 0.9555, Top-10: 0.9719
Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau huan luyen mo hinh PhoBERT...
🚀 Device: cuda
Số lượng nhãn: 68
Ví dụ 10 nhãn đầu: ['Bệnh Alzheimer', 'Bệnh CreutzfeldtJakob', 'Bệnh Parkinson', 'Bệnh rối loạn tâm thần ở tuổi già', 'Bệnh sợ khoảng trống', 'Bệnh tâm thần', 'Chán ăn tâm thần', 'Chứng háu ăn tâm thần', 'Chứng nghiện', 'Chứng sợ không gian hẹp (Claustrophobia)']
pos_weight (top 10): tensor([193.7714, 116.5345, 102.2879, 173.7949, 234.0690, 108.9516,  55.3388,
        122.9454,  36.2514,  55.3388], device='cuda:0')

Epoch 1/20: 100%|████████████████████████████████████████████████████████| 214/214 [18:50<00:00,  5.28s/it, loss=0.844]
📊 Epoch 1: Loss=1.3503, Val F1 micro=0.0753, macro=0.1117
   Top-1: 0.1691, Top-5: 0.4168, Top-10: 0.5927
⭐ Saved best model (F1 micro=0.0753)
Epoch 2/20: 100%|████████████████████████████████████████████████████████| 214/214 [18:53<00:00,  5.30s/it, loss=0.407]
📊 Epoch 2: Loss=0.9438, Val F1 micro=0.1227, macro=0.1347
   Top-1: 0.4312, Top-5: 0.7118, Top-10: 0.8015
⭐ Saved best model (F1 micro=0.1227)
Epoch 3/20: 100%|████████████████████████████████████████████████████████| 214/214 [18:51<00:00,  5.29s/it, loss=0.237]
📊 Epoch 3: Loss=0.4737, Val F1 micro=0.2776, macro=0.2602
   Top-1: 0.5530, Top-5: 0.8617, Top-10: 0.9336
⭐ Saved best model (F1 micro=0.2776)
Epoch 4/20: 100%|████████████████████████████████████████████████████████| 214/214 [19:06<00:00,  5.36s/it, loss=0.129]
📊 Epoch 4: Loss=0.2904, Val F1 micro=0.3116, macro=0.3159
   Top-1: 0.6105, Top-5: 0.8604, Top-10: 0.9452
⭐ Saved best model (F1 micro=0.3116)
Epoch 5/20: 100%|███████████████████████████████████████████████████████| 214/214 [18:53<00:00,  5.29s/it, loss=0.0913]
📊 Epoch 5: Loss=0.2072, Val F1 micro=0.3940, macro=0.3826
   Top-1: 0.6762, Top-5: 0.9172, Top-10: 0.9610
⭐ Saved best model (F1 micro=0.3940)
Epoch 6/20: 100%|█████████████████████████████████████████████████████████| 214/214 [19:14<00:00,  5.40s/it, loss=0.22]
📊 Epoch 6: Loss=0.1500, Val F1 micro=0.4825, macro=0.4788
   Top-1: 0.7180, Top-5: 0.9268, Top-10: 0.9692
⭐ Saved best model (F1 micro=0.4825)
Epoch 7/20: 100%|████████████████████████████████████████████████████████| 214/214 [20:54<00:00,  5.86s/it, loss=0.342]
📊 Epoch 7: Loss=0.1170, Val F1 micro=0.5396, macro=0.5434
   Top-1: 0.7084, Top-5: 0.9398, Top-10: 0.9713
⭐ Saved best model (F1 micro=0.5396)
Epoch 8/20: 100%|███████████████████████████████████████████████████████| 214/214 [23:10<00:00,  6.50s/it, loss=0.0083]
📊 Epoch 8: Loss=0.0859, Val F1 micro=0.6001, macro=0.6100
   Top-1: 0.7536, Top-5: 0.9493, Top-10: 0.9726
⭐ Saved best model (F1 micro=0.6001)
Epoch 9/20: 100%|████████████████████████████████████████████████████████| 214/214 [21:57<00:00,  6.15s/it, loss=0.108]
📊 Epoch 9: Loss=0.0679, Val F1 micro=0.6504, macro=0.6563
   Top-1: 0.7673, Top-5: 0.9562, Top-10: 0.9767
⭐ Saved best model (F1 micro=0.6504)
Epoch 10/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00871]
📊 Epoch 10: Loss=0.0535, Val F1 micro=0.6789, macro=0.6894
   Top-1: 0.7769, Top-5: 0.9487, Top-10: 0.9733
⭐ Saved best model (F1 micro=0.6789)
Epoch 11/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00162]
📊 Epoch 11: Loss=0.0447, Val F1 micro=0.7288, macro=0.7334
   Top-1: 0.7892, Top-5: 0.9562, Top-10: 0.9726
⭐ Saved best model (F1 micro=0.7288)
Epoch 12/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.0252]
📊 Epoch 12: Loss=0.0372, Val F1 micro=0.7309, macro=0.7321
   Top-1: 0.7967, Top-5: 0.9589, Top-10: 0.9699
⭐ Saved best model (F1 micro=0.7309)
Epoch 13/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.0661]
📊 Epoch 13: Loss=0.0301, Val F1 micro=0.7408, macro=0.7400
   Top-1: 0.8049, Top-5: 0.9603, Top-10: 0.9740
⭐ Saved best model (F1 micro=0.7408)
Epoch 14/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00537]
📊 Epoch 14: Loss=0.0254, Val F1 micro=0.7623, macro=0.7693
   Top-1: 0.8152, Top-5: 0.9576, Top-10: 0.9706
⭐ Saved best model (F1 micro=0.7623)
Epoch 15/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00416]
📊 Epoch 15: Loss=0.0206, Val F1 micro=0.7748, macro=0.7717
   Top-1: 0.8159, Top-5: 0.9514, Top-10: 0.9713
⭐ Saved best model (F1 micro=0.7748)
Epoch 16/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.0205]
📊 Epoch 16: Loss=0.0185, Val F1 micro=0.7984, macro=0.8027
   Top-1: 0.8220, Top-5: 0.9555, Top-10: 0.9699
⭐ Saved best model (F1 micro=0.7984)
Epoch 17/20: 100%|████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.000783]
📊 Epoch 17: Loss=0.0171, Val F1 micro=0.7970, macro=0.8080
   Top-1: 0.8241, Top-5: 0.9582, Top-10: 0.9713
Epoch 18/20: 100%|██████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.0135]
📊 Epoch 18: Loss=0.0148, Val F1 micro=0.7969, macro=0.8047
   Top-1: 0.8241, Top-5: 0.9569, Top-10: 0.9726
Epoch 19/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00212]
📊 Epoch 19: Loss=0.0118, Val F1 micro=0.7971, macro=0.8069
   Top-1: 0.8248, Top-5: 0.9548, Top-10: 0.9713
Epoch 20/20: 100%|█████████████████████████████████████████████████████| 214/214 [19:01<00:00,  5.33s/it, loss=0.00575]
📊 Epoch 20: Loss=0.0119, Val F1 micro=0.8024, macro=0.8106
   Top-1: 0.8234, Top-5: 0.9555, Top-10: 0.9719
⭐ Saved best model (F1 micro=0.8024)
Training completed.
. .


# ----------------Test---------------------------------------#
Loaded 68 labels
Test samples: 1461
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2648.73it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base

# -------Result with test file ------- #
✅ Test F1 micro: 0.7896, macro: 0.7968
   Top-1 accuracy: 0.8015
   Top-5 accuracy: 0.9439
   Top-10 accuracy: 0.9706
   Top-20 accuracy: 0.9836

# -------Result with random exsample-------#
📝 Dự đoán mẫu (top 5):

Text: Tôi cảm thấy buồn bã, mất ngủ, không muốn gặp ai, nghĩ đến cái chết....
Top 5: [('Rối loạn trầm cảm', 0.9936360120773315), ('Khác', 0.7801176309585571), ('Rối loạn giấc ngủ (Hypersomnia)', 0.21289560198783875), ('Rối loạn loạn thần', 0.003582380246371031), ('Rối loạn tâm lý', 0.001835949020460248)]

Text: Tôi sợ đi thang máy, mỗi lần thấy nhện là hoảng hốt, tay run....
Top 5: [('Khác', 0.9121453166007996), ('Chứng sợ không gian hẹp (Claustrophobia)', 0.764439046382904), ('Hội chứng Fregoli', 0.0037611129228025675), ('Hoang tưởng', 0.0006425243918783963), ('Bệnh sợ khoảng trống', 0.0003956558939535171)]
