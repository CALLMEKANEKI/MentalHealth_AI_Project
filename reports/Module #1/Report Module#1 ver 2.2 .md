# Module#1 Ver 2.2
# Update: pos_weight, new caculate batch_loss


# ----------------Training-----------------------#
# Best model: Epoch 
Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau huan luyen mo hinh PhoBERT...
📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Device: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Emotion  train=22022 | valid=2742
✅ Hate     train=5163 | valid=738

📊 Train total: 27185 | Val total: 3480
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 4769.47it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/20: 100%|██████████████████████████████████| 2266/2266 [21:06<00:00,  1.79it/s, E=0.85, H=0.19, loss=0.5170]
📊 Epoch 1: Loss=1.0359
   Emotion → F1 micro=0.2107 | F1 macro=0.1641
   Hate    → Acc=0.6504
   Combined score: 0.3866
⭐ Đã lưu best model (score=0.3866)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/20: 100%|██████████████████████████████████| 2266/2266 [20:43<00:00,  1.82it/s, E=0.68, H=0.80, loss=0.3376]
📊 Epoch 2: Loss=0.8156
   Emotion → F1 micro=0.3368 | F1 macro=0.3325
   Hate    → Acc=0.7290
   Combined score: 0.4937
⭐ Đã lưu best model (score=0.4937)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/20: 100%|██████████████████████████████████| 2266/2266 [20:51<00:00,  1.81it/s, E=0.41, H=0.05, loss=0.2313]
📊 Epoch 3: Loss=0.6337
   Emotion → F1 micro=0.3836 | F1 macro=0.3693
   Hate    → Acc=0.7425
   Combined score: 0.5272
⭐ Đã lưu best model (score=0.5272)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/20: 100%|██████████████████████████████████| 2266/2266 [20:58<00:00,  1.80it/s, E=0.45, H=1.15, loss=0.6640]
📊 Epoch 4: Loss=0.5533
   Emotion → F1 micro=0.4134 | F1 macro=0.4080
   Hate    → Acc=0.7832
   Combined score: 0.5613
⭐ Đã lưu best model (score=0.5613)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/20: 100%|██████████████████████████████████| 2266/2266 [20:48<00:00,  1.82it/s, E=0.27, H=0.04, loss=0.1980]
📊 Epoch 5: Loss=0.4520
   Emotion → F1 micro=0.4357 | F1 macro=0.4365
   Hate    → Acc=0.7818
   Combined score: 0.5741
⭐ Đã lưu best model (score=0.5741)

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/20: 100%|██████████████████████████████████| 2266/2266 [20:40<00:00,  1.83it/s, E=0.58, H=0.00, loss=0.4064]
📊 Epoch 6: Loss=0.3826
   Emotion → F1 micro=0.4424 | F1 macro=0.4430
   Hate    → Acc=0.7900
   Combined score: 0.5814
⭐ Đã lưu best model (score=0.5814)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/20: 100%|██████████████████████████████████| 2266/2266 [20:39<00:00,  1.83it/s, E=0.44, H=0.00, loss=0.3064]
📊 Epoch 7: Loss=0.3302
   Emotion → F1 micro=0.4533 | F1 macro=0.4560
   Hate    → Acc=0.7995
   Combined score: 0.5918
⭐ Đã lưu best model (score=0.5918)

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/20: 100%|██████████████████████████████████| 2266/2266 [20:38<00:00,  1.83it/s, E=0.59, H=0.00, loss=0.4104]
📊 Epoch 8: Loss=0.2874
   Emotion → F1 micro=0.4681 | F1 macro=0.4760
   Hate    → Acc=0.7846
   Combined score: 0.5947
⭐ Đã lưu best model (score=0.5947)

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/20: 100%|██████████████████████████████████| 2266/2266 [20:45<00:00,  1.82it/s, E=0.63, H=0.00, loss=0.4412]
📊 Epoch 9: Loss=0.2568
   Emotion → F1 micro=0.4860 | F1 macro=0.4863
   Hate    → Acc=0.8035
   Combined score: 0.6130
⭐ Đã lưu best model (score=0.6130)

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/20: 100%|█████████████████████████████████| 2266/2266 [20:40<00:00,  1.83it/s, E=0.22, H=0.00, loss=0.1550]
📊 Epoch 10: Loss=0.2311
   Emotion → F1 micro=0.4945 | F1 macro=0.4935
   Hate    → Acc=0.7900
   Combined score: 0.6127

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/20: 100%|█████████████████████████████████| 2266/2266 [20:40<00:00,  1.83it/s, E=0.08, H=0.00, loss=0.0548]
📊 Epoch 11: Loss=0.2142
   Emotion → F1 micro=0.4943 | F1 macro=0.4943
   Hate    → Acc=0.7913
   Combined score: 0.6131
⭐ Đã lưu best model (score=0.6131)

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/20: 100%|█████████████████████████████████| 2266/2266 [20:39<00:00,  1.83it/s, E=0.21, H=0.00, loss=0.1437]
📊 Epoch 12: Loss=0.1955
   Emotion → F1 micro=0.4972 | F1 macro=0.4963
   Hate    → Acc=0.7967
   Combined score: 0.6170
⭐ Đã lưu best model (score=0.6170)

--- Chiến lược Epoch 13: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 13/20: 100%|█████████████████████████████████| 2266/2266 [20:38<00:00,  1.83it/s, E=0.13, H=0.01, loss=0.1043]
📊 Epoch 13: Loss=0.2062
   Emotion → F1 micro=0.4977 | F1 macro=0.4951
   Hate    → Acc=0.8049
   Combined score: 0.6206
⭐ Đã lưu best model (score=0.6206)

--- Chiến lược Epoch 14: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 14/20: 100%|█████████████████████████████████| 2266/2266 [20:38<00:00,  1.83it/s, E=0.27, H=0.13, loss=0.2454]
📊 Epoch 14: Loss=0.1933
   Emotion → F1 micro=0.5051 | F1 macro=0.5037
   Hate    → Acc=0.7927
   Combined score: 0.6201

--- Chiến lược Epoch 15: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 15/20: 100%|█████████████████████████████████| 2266/2266 [20:38<00:00,  1.83it/s, E=0.13, H=0.00, loss=0.1045]
📊 Epoch 15: Loss=0.1778
   Emotion → F1 micro=0.5048 | F1 macro=0.5015
   Hate    → Acc=0.7940
   Combined score: 0.6205

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/20: 100%|█████████████████████████████████| 2266/2266 [20:39<00:00,  1.83it/s, E=0.12, H=0.00, loss=0.0941]
📊 Epoch 16: Loss=0.1668
   Emotion → F1 micro=0.5096 | F1 macro=0.5065
   Hate    → Acc=0.7940
   Combined score: 0.6234
⭐ Đã lưu best model (score=0.6234)

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/20: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.14, H=0.00, loss=0.1103]
📊 Epoch 17: Loss=0.1590
   Emotion → F1 micro=0.5099 | F1 macro=0.5063
   Hate    → Acc=0.7954
   Combined score: 0.6241
⭐ Đã lưu best model (score=0.6241)

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/20: 100%|█████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.61, H=0.00, loss=0.4870]
📊 Epoch 18: Loss=0.1501
   Emotion → F1 micro=0.5088 | F1 macro=0.5066
   Hate    → Acc=0.7927
   Combined score: 0.6223

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/20: 100%|█████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.26, H=0.00, loss=0.2046]
📊 Epoch 19: Loss=0.1458
   Emotion → F1 micro=0.5079 | F1 macro=0.5032
   Hate    → Acc=0.8022
   Combined score: 0.6256
⭐ Đã lưu best model (score=0.6256)

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/20: 100%|█████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.05, H=0.00, loss=0.0413]
📊 Epoch 20: Loss=0.1392
   Emotion → F1 micro=0.5081 | F1 macro=0.5046
   Hate    → Acc=0.8076
   Combined score: 0.6279
⭐ Đã lưu best model (score=0.6279)

✅ Training hoàn tất. Best combined score: 0.6279
Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .


Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau kiem tra mo hinh PhoBERT...
📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Đánh giá trên: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Loaded tuned thresholds từ: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\checkpoints\Module#1 ver 2.2\emotion_thresholds.json
✅ Emotion test: 2751 | Hate test: 1475
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 9191.35it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📊 Đang dự đoán...

==================== EMOTION REPORT (Multi-label 28 nhãn) ====================

📊 So sánh threshold:
   @threshold=0.5:    F1 micro=0.5287 | F1 macro=0.5187
   @tuned threshold:  F1 micro=0.5179 | F1 macro=0.5185  (+-0.0108 micro | +-0.0002 macro)

⚠️  Nhãn F1 thấp nhất (bottom 10):
  disapproval          F1=0.268 thresh=0.15 █████
  excitement           F1=0.329 thresh=0.20 ██████
  confusion            F1=0.343 thresh=0.80 ██████
  realization          F1=0.367 thresh=0.20 ███████
  neutral              F1=0.369 thresh=0.60 ███████
  disappointment       F1=0.380 thresh=0.55 ███████
  desire               F1=0.398 thresh=0.45 ███████
  admiration           F1=0.425 thresh=0.40 ████████
  nervousness          F1=0.435 thresh=0.25 ████████
  curiosity            F1=0.454 thresh=0.65 █████████

✅ Nhãn F1 cao nhất (top 10):
  surprise             F1=0.551 thresh=0.80 ███████████
  love                 F1=0.563 thresh=0.90 ███████████
  pride                F1=0.571 thresh=0.90 ███████████
  joy                  F1=0.607 thresh=0.80 ████████████
  optimism             F1=0.628 thresh=0.40 ████████████
  remorse              F1=0.671 thresh=0.75 █████████████
  sadness              F1=0.709 thresh=0.90 ██████████████
  fear                 F1=0.737 thresh=0.80 ██████████████
  embarrassment        F1=0.808 thresh=0.55 ████████████████
  gratitude            F1=0.814 thresh=0.85 ████████████████

==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.89      0.86      0.88       599
   Offensive       0.74      0.77      0.75       496
        Hate       0.80      0.79      0.80       380

    accuracy                           0.81      1475
   macro avg       0.81      0.81      0.81      1475
weighted avg       0.82      0.81      0.81      1475

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .