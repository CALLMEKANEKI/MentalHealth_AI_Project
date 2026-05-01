## Module 2 ver 1.3: Cải thiện pos_weight, undersample nhãn "Khác", bổ sung Per-label F1 Report
## ---Data---##
# Total_label: 68
# Min_sample: 30
# ABSOLUTE_MIN = 10. Loại bỏ các topic có số dòng nhỏ hơn 10

## ---Config---##
# Total epoch: 20
# BATCH_SIZE = 32
# MAX_LEN = 128
# LR_BACKBONE = 2e-5
# LR_HEAD = 5e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EARLY_STOP_PATIENCE = 5
# GRAD_CLIP = 1.0
# WEIGHT_DECAY = 0.01

# Best model: Epoch 20 with Val F1 micro=0.8260, macro=0.8337. Top-1: 0.8438, Top-5: 0.9466, Top-10: 0.9655
Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau huan luyen mo hinh PhoBERT...
🚀 Device: cuda
Số lượng nhãn: 68
Ví dụ 10 nhãn đầu: ['Bệnh Alzheimer', 'Bệnh CreutzfeldtJakob', 'Bệnh Parkinson', 'Bệnh rối loạn tâm thần ở tuổi già', 'Bệnh sợ khoảng trống', 'Bệnh tâm thần', 'Chán ăn tâm thần', 'Chứng háu ăn tâm thần', 'Chứng nghiện', 'Chứng sợ không gian hẹp (Claustrophobia)']
pos_weight: min=8.7, max=20.0, mean=19.8

Epoch 1/20: 100%|████████████████████████████████████████████████████████| 186/186 [16:32<00:00,  5.34s/it, loss=0.641]
📊 Epoch 1: Loss=0.7016, Val F1 micro=0.1422, macro=0.0304
   Top-1: 0.1209, Top-5: 0.3336, Top-10: 0.4827
⭐ Saved best model (F1 micro=0.1422)
Epoch 2/20: 100%|████████████████████████████████████████████████████████| 186/186 [16:33<00:00,  5.34s/it, loss=0.217]
📊 Epoch 2: Loss=0.4312, Val F1 micro=0.3901, macro=0.3720
   Top-1: 0.4874, Top-5: 0.8462, Top-10: 0.9176
⭐ Saved best model (F1 micro=0.3901)
Epoch 3/20: 100%|████████████████████████████████████████████████████████| 186/186 [16:32<00:00,  5.33s/it, loss=0.215]
📊 Epoch 3: Loss=0.2087, Val F1 micro=0.4750, macro=0.4990
   Top-1: 0.6601, Top-5: 0.9137, Top-10: 0.9513
⭐ Saved best model (F1 micro=0.4750)
Epoch 4/20: 100%|███████████████████████████████████████████████████████| 186/186 [16:32<00:00,  5.33s/it, loss=0.0678]
📊 Epoch 4: Loss=0.1360, Val F1 micro=0.5533, macro=0.6033
   Top-1: 0.7261, Top-5: 0.9364, Top-10: 0.9615
⭐ Saved best model (F1 micro=0.5533)
Epoch 5/20: 100%|████████████████████████████████████████████████████████| 186/186 [17:13<00:00,  5.56s/it, loss=0.109]
📊 Epoch 5: Loss=0.0954, Val F1 micro=0.5932, macro=0.6292
   Top-1: 0.7645, Top-5: 0.9403, Top-10: 0.9662
⭐ Saved best model (F1 micro=0.5932)
Epoch 6/20: 100%|███████████████████████████████████████████████████████| 186/186 [16:30<00:00,  5.33s/it, loss=0.0472]
📊 Epoch 6: Loss=0.0706, Val F1 micro=0.6520, macro=0.6824
   Top-1: 0.7857, Top-5: 0.9396, Top-10: 0.9623
⭐ Saved best model (F1 micro=0.6520)
Epoch 7/20: 100%|███████████████████████████████████████████████████████| 186/186 [16:30<00:00,  5.33s/it, loss=0.0312]
📊 Epoch 7: Loss=0.0537, Val F1 micro=0.6837, macro=0.7153
   Top-1: 0.7920, Top-5: 0.9364, Top-10: 0.9639
⭐ Saved best model (F1 micro=0.6837)
Epoch 8/20: 100%|███████████████████████████████████████████████████████| 186/186 [17:22<00:00,  5.60s/it, loss=0.0448]
📊 Epoch 8: Loss=0.0436, Val F1 micro=0.7162, macro=0.7493
   Top-1: 0.7975, Top-5: 0.9451, Top-10: 0.9670
⭐ Saved best model (F1 micro=0.7162)
Epoch 9/20: 100%|████████████████████████████████████████████████████████| 186/186 [19:52<00:00,  6.41s/it, loss=0.056]
📊 Epoch 9: Loss=0.0351, Val F1 micro=0.7331, macro=0.7630
   Top-1: 0.8100, Top-5: 0.9451, Top-10: 0.9710
⭐ Saved best model (F1 micro=0.7331)
Epoch 10/20: 100%|██████████████████████████████████████████████████████| 186/186 [23:23<00:00,  7.55s/it, loss=0.0274]
📊 Epoch 10: Loss=0.0272, Val F1 micro=0.7659, macro=0.7887
   Top-1: 0.8179, Top-5: 0.9466, Top-10: 0.9710
⭐ Saved best model (F1 micro=0.7659)
Epoch 11/20: 100%|██████████████████████████████████████████████████████| 186/186 [21:08<00:00,  6.82s/it, loss=0.0224]
📊 Epoch 11: Loss=0.0232, Val F1 micro=0.7771, macro=0.7929
   Top-1: 0.8195, Top-5: 0.9443, Top-10: 0.9702
⭐ Saved best model (F1 micro=0.7771)
Epoch 12/20: 100%|██████████████████████████████████████████████████████| 186/186 [17:27<00:00,  5.63s/it, loss=0.0133]
📊 Epoch 12: Loss=0.0186, Val F1 micro=0.7877, macro=0.7927
   Top-1: 0.8203, Top-5: 0.9443, Top-10: 0.9662
⭐ Saved best model (F1 micro=0.7877)
Epoch 13/20: 100%|██████████████████████████████████████████████████████| 186/186 [17:42<00:00,  5.71s/it, loss=0.0154]
📊 Epoch 13: Loss=0.0159, Val F1 micro=0.7973, macro=0.8083
   Top-1: 0.8359, Top-5: 0.9474, Top-10: 0.9694
⭐ Saved best model (F1 micro=0.7973)
Epoch 14/20: 100%|█████████████████████████████████████████████████████| 186/186 [16:40<00:00,  5.38s/it, loss=0.00562]
📊 Epoch 14: Loss=0.0132, Val F1 micro=0.8088, macro=0.8155
   Top-1: 0.8328, Top-5: 0.9435, Top-10: 0.9694
⭐ Saved best model (F1 micro=0.8088)
Epoch 15/20: 100%|███████████████████████████████████████████████████████| 186/186 [16:39<00:00,  5.38s/it, loss=0.037]
📊 Epoch 15: Loss=0.0113, Val F1 micro=0.8141, macro=0.8237
   Top-1: 0.8273, Top-5: 0.9427, Top-10: 0.9655
⭐ Saved best model (F1 micro=0.8141)
Epoch 16/20: 100%|█████████████████████████████████████████████████████| 186/186 [16:42<00:00,  5.39s/it, loss=0.00787]
📊 Epoch 16: Loss=0.0095, Val F1 micro=0.8193, macro=0.8217
   Top-1: 0.8414, Top-5: 0.9435, Top-10: 0.9639
⭐ Saved best model (F1 micro=0.8193)
Epoch 17/20: 100%|██████████████████████████████████████████████████████| 186/186 [19:50<00:00,  6.40s/it, loss=0.0106]
📊 Epoch 17: Loss=0.0090, Val F1 micro=0.8230, macro=0.8337
   Top-1: 0.8430, Top-5: 0.9458, Top-10: 0.9662
⭐ Saved best model (F1 micro=0.8230)
Epoch 18/20: 100%|█████████████████████████████████████████████████████| 186/186 [17:44<00:00,  5.72s/it, loss=0.00746]
📊 Epoch 18: Loss=0.0075, Val F1 micro=0.8191, macro=0.8260
   Top-1: 0.8430, Top-5: 0.9443, Top-10: 0.9655
Epoch 19/20: 100%|█████████████████████████████████████████████████████| 186/186 [18:39<00:00,  6.02s/it, loss=0.00363]
📊 Epoch 19: Loss=0.0068, Val F1 micro=0.8224, macro=0.8272
   Top-1: 0.8414, Top-5: 0.9443, Top-10: 0.9655
Epoch 20/20: 100%|███████████████████████████████████████████████████████| 186/186 [18:14<00:00,  5.88s/it, loss=0.013]
📊 Epoch 20: Loss=0.0063, Val F1 micro=0.8260, macro=0.8337
   Top-1: 0.8438, Top-5: 0.9466, Top-10: 0.9655
⭐ Saved best model (F1 micro=0.8260)



# ----------------Test---------------------------------------#
Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau huan luyen mo hinh PhoBERT...
Loaded 68 labels
Test samples: 1275

✅ Test F1 micro: 0.8260, macro: 0.8339
   Top-1 accuracy: 0.8243
   Top-5 accuracy: 0.9427
   Top-10 accuracy: 0.9647
   Top-20 accuracy: 0.9796

=== PER-LABEL F1 REPORT ===

⚠️  Nhãn có F1 thấp nhất (bottom 10):
  Rối loạn tâm lý                               F1=0.364
  Rối loạn tâm thần                             F1=0.455
  Rối loạn hệ thần kinh thực vật ở trẻ em       F1=0.600
  Rối loạn trầm cảm                             F1=0.606
  Rối loạn nhân cách phụ thuộc (DPD)            F1=0.615
  Rối loạn hành vi                              F1=0.632
  Rối loạn dạng cơ thể                          F1=0.667
  Rối loạn lo âu lan tỏa (GAD)                  F1=0.683
  Rối loạn giấc ngủ (Hypersomnia)               F1=0.688
  Khác                                          F1=0.693

✅ Nhãn có F1 cao nhất (top 10):
  Chán ăn tâm thần                              F1=0.945
  Rối loạn khí sắc                              F1=0.947
  Rối loạn chức năng tình dục                   F1=0.952
  Nghiện mạng xã hội                            F1=0.953
  Bệnh Parkinson                                F1=0.966
  Ảo Thanh                                      F1=0.980
  Bệnh CreutzfeldtJakob                         F1=1.000
  Hội chứng Alice in Wonderland                 F1=1.000
  Hội chứng Hoarding                            F1=1.000
  Rối loạn cường dương                          F1=1.000

🔍 'Khác' riêng: Precision=0.649 | Recall=0.742 | F1=0.693

📝 Dự đoán mẫu (top 5):

Text: Tôi cảm thấy buồn bã, mất ngủ, không muốn gặp ai, nghĩ đến cái chết....
Top 5: [('Rối loạn trầm cảm', 0.9880306124687195), ('Khác', 0.13032162189483643), ('Rối loạn điều chỉnh (AD)', 0.04688921943306923), ('Rối loạn giấc ngủ (Hypersomnia)', 0.045859165489673615), ('Rối loạn loạn thần', 0.014025693759322166)]

Text: Tôi sợ đi thang máy, mỗi lần thấy nhện là hoảng hốt, tay run....
Top 5: [('Khác', 0.8902126550674438), ('Chứng sợ không gian hẹp (Claustrophobia)', 0.2644987106323242), ('Rối loạn hoảng sợ (Panic disorder)', 0.0597829669713974), ('Rối loạn dạng cơ thể', 0.007130834739655256), ('Hội chứng Fregoli', 0.003605922218412161)]
