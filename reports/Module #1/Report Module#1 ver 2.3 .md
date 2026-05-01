# Module#1 Ver 2.3


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
   Sampler weight — Emotion: 1.23x | Hate: 5.27x
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 5106.37it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
   pos_weight — min=4.9 | max=20.0 | mean=16.6

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/30: 100%|██████████████████████████████████| 2266/2266 [20:39<00:00,  1.83it/s, E=1.12, H=0.81, loss=0.9633]
📊 Epoch 1: Loss=1.0779 | NaN batches skipped=0
   Emotion → F1 micro=0.2281 | F1 macro=0.1660
   Hate    → Acc=0.6911
   Combined score: 0.4133
⭐ Đã lưu best model (score=0.4133)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/30: 100%|██████████████████████████████████| 2266/2266 [20:33<00:00,  1.84it/s, E=0.67, H=0.76, loss=0.7120]
📊 Epoch 2: Loss=0.8147 | NaN batches skipped=0
   Emotion → F1 micro=0.3171 | F1 macro=0.3033
   Hate    → Acc=0.7588
   Combined score: 0.4938
⭐ Đã lưu best model (score=0.4938)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/30: 100%|██████████████████████████████████| 2266/2266 [20:34<00:00,  1.84it/s, E=0.56, H=0.30, loss=0.4279]
📊 Epoch 3: Loss=0.6260 | NaN batches skipped=0
   Emotion → F1 micro=0.3580 | F1 macro=0.3420
   Hate    → Acc=0.7764
   Combined score: 0.5254
⭐ Đã lưu best model (score=0.5254)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/30: 100%|██████████████████████████████████| 2266/2266 [20:34<00:00,  1.83it/s, E=0.33, H=0.36, loss=0.3361]
📊 Epoch 4: Loss=0.5835 | NaN batches skipped=0
   Emotion → F1 micro=0.4059 | F1 macro=0.4012
   Hate    → Acc=0.7927
   Combined score: 0.5606
⭐ Đã lưu best model (score=0.5606)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/30: 100%|██████████████████████████████████| 2266/2266 [20:35<00:00,  1.83it/s, E=0.26, H=0.31, loss=0.2731]
📊 Epoch 5: Loss=0.5225 | NaN batches skipped=0
   Emotion → F1 micro=0.4390 | F1 macro=0.4407
   Hate    → Acc=0.7995
   Combined score: 0.5832
⭐ Đã lưu best model (score=0.5832)

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/30: 100%|██████████████████████████████████| 2266/2266 [20:35<00:00,  1.83it/s, E=0.47, H=0.30, loss=0.4150]
📊 Epoch 6: Loss=0.4783 | NaN batches skipped=0
   Emotion → F1 micro=0.4717 | F1 macro=0.4742
   Hate    → Acc=0.7873
   Combined score: 0.5979
⭐ Đã lưu best model (score=0.5979)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/30: 100%|██████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.53, H=0.32, loss=0.4687]
📊 Epoch 7: Loss=0.4405 | NaN batches skipped=0
   Emotion → F1 micro=0.4707 | F1 macro=0.4723
   Hate    → Acc=0.7832
   Combined score: 0.5957

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/30: 100%|██████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=1.16, H=0.30, loss=0.9027]
📊 Epoch 8: Loss=0.4135 | NaN batches skipped=0
   Emotion → F1 micro=0.4672 | F1 macro=0.4685
   Hate    → Acc=0.7710
   Combined score: 0.5887

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/30: 100%|██████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.18, H=0.35, loss=0.2325]
📊 Epoch 9: Loss=0.3986 | NaN batches skipped=0
   Emotion → F1 micro=0.4908 | F1 macro=0.4902
   Hate    → Acc=0.7805
   Combined score: 0.6067
⭐ Đã lưu best model (score=0.6067)

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.28, H=0.00, loss=0.1926]
📊 Epoch 10: Loss=0.3688 | NaN batches skipped=0
   Emotion → F1 micro=0.4869 | F1 macro=0.4923
   Hate    → Acc=0.7940
   Combined score: 0.6098
⭐ Đã lưu best model (score=0.6098)

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.11, H=0.30, loss=0.1676]
📊 Epoch 11: Loss=0.3524 | NaN batches skipped=0
   Emotion → F1 micro=0.4973 | F1 macro=0.4948
   Hate    → Acc=0.8008
   Combined score: 0.6187
⭐ Đã lưu best model (score=0.6187)

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/30: 100%|█████████████████████████████████| 2266/2266 [20:35<00:00,  1.83it/s, E=0.19, H=0.32, loss=0.2302]
📊 Epoch 12: Loss=0.3359 | NaN batches skipped=0
   Emotion → F1 micro=0.4996 | F1 macro=0.5016
   Hate    → Acc=0.7981
   Combined score: 0.6190
⭐ Đã lưu best model (score=0.6190)

--- Chiến lược Epoch 13: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 13/30: 100%|█████████████████████████████████| 2266/2266 [20:40<00:00,  1.83it/s, E=0.10, H=0.32, loss=0.1653]
📊 Epoch 13: Loss=0.3203 | NaN batches skipped=0
   Emotion → F1 micro=0.5012 | F1 macro=0.5010
   Hate    → Acc=0.7995
   Combined score: 0.6205
⭐ Đã lưu best model (score=0.6205)

--- Chiến lược Epoch 14: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 14/30: 100%|█████████████████████████████████| 2266/2266 [20:42<00:00,  1.82it/s, E=0.25, H=0.43, loss=0.3050]
📊 Epoch 14: Loss=0.3073 | NaN batches skipped=0
   Emotion → F1 micro=0.5045 | F1 macro=0.5019
   Hate    → Acc=0.7859
   Combined score: 0.6171

--- Chiến lược Epoch 15: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 15/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.00, H=0.37, loss=0.1112]
📊 Epoch 15: Loss=0.2943 | NaN batches skipped=0
   Emotion → F1 micro=0.5044 | F1 macro=0.5000
   Hate    → Acc=0.7886
   Combined score: 0.6181

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.15, H=0.30, loss=0.1842]
📊 Epoch 16: Loss=0.2831 | NaN batches skipped=0
   Emotion → F1 micro=0.5083 | F1 macro=0.5082
   Hate    → Acc=0.7900
   Combined score: 0.6210
⭐ Đã lưu best model (score=0.6210)

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.16, H=0.30, loss=0.1904]
📊 Epoch 17: Loss=0.2704 | NaN batches skipped=0
   Emotion → F1 micro=0.5095 | F1 macro=0.5073
   Hate    → Acc=0.7846
   Combined score: 0.6195

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/30: 100%|█████████████████████████████████| 2266/2266 [20:35<00:00,  1.83it/s, E=0.11, H=0.35, loss=0.1604]
📊 Epoch 18: Loss=0.2582 | NaN batches skipped=0
   Emotion → F1 micro=0.5041 | F1 macro=0.5022
   Hate    → Acc=0.7981
   Combined score: 0.6217
⭐ Đã lưu best model (score=0.6217)

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/30: 100%|█████████████████████████████████| 2266/2266 [20:33<00:00,  1.84it/s, E=0.32, H=0.32, loss=0.3239]
📊 Epoch 19: Loss=0.2471 | NaN batches skipped=0
   Emotion → F1 micro=0.5081 | F1 macro=0.5025
   Hate    → Acc=0.7967
   Combined score: 0.6236
⭐ Đã lưu best model (score=0.6236)

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/30: 100%|█████████████████████████████████| 2266/2266 [20:34<00:00,  1.84it/s, E=0.11, H=0.30, loss=0.1468]
📊 Epoch 20: Loss=0.2433 | NaN batches skipped=0
   Emotion → F1 micro=0.5051 | F1 macro=0.4994
   Hate    → Acc=0.7846
   Combined score: 0.6169

--- Chiến lược Epoch 21: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 21/30: 100%|█████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.28, H=0.29, loss=0.2825]
📊 Epoch 21: Loss=0.2374 | NaN batches skipped=0
   Emotion → F1 micro=0.4974 | F1 macro=0.4895
   Hate    → Acc=0.7805
   Combined score: 0.6106

--- Chiến lược Epoch 22: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 22/30: 100%|█████████████████████████████████| 2266/2266 [20:37<00:00,  1.83it/s, E=0.19, H=0.33, loss=0.2147]
📊 Epoch 22: Loss=0.2305 | NaN batches skipped=0
   Emotion → F1 micro=0.5082 | F1 macro=0.5084
   Hate    → Acc=0.7886
   Combined score: 0.6203

--- Chiến lược Epoch 23: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 23/30: 100%|█████████████████████████████████| 2266/2266 [20:36<00:00,  1.83it/s, E=0.05, H=0.29, loss=0.0997]
📊 Epoch 23: Loss=0.2231 | NaN batches skipped=0
   Emotion → F1 micro=0.5073 | F1 macro=0.5053
   Hate    → Acc=0.7859
   Combined score: 0.6187

--- Chiến lược Epoch 24: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 24/30: 100%|█████████████████████████████████| 2266/2266 [20:38<00:00,  1.83it/s, E=0.09, H=0.30, loss=0.1342]
📊 Epoch 24: Loss=0.2135 | NaN batches skipped=0
   Emotion → F1 micro=0.5081 | F1 macro=0.5082
   Hate    → Acc=0.7859
   Combined score: 0.6192
🛑 Early stopping tại Epoch 24

✅ Training hoàn tất. Best combined score: 0.6236
Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .

Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau kiem tra mo hinh PhoBERT...
📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Đánh giá trên: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Loaded tuned thresholds từ: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\checkpoints\Module#1 ver 2.3\emotion_thresholds.json
✅ Emotion test: 2751 | Hate test: 1475
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8047.46it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📊 Đang dự đoán...

==================== EMOTION REPORT (Multi-label 28 nhãn) ====================

📊 So sánh threshold:
   @threshold=0.5:    F1 micro=0.5161 | F1 macro=0.5048
   @tuned threshold:  F1 micro=0.5126 | F1 macro=0.5129  (+-0.0036 micro | +0.0082 macro)

⚠️  Nhãn F1 thấp nhất (bottom 10):
  disapproval          F1=0.277 thresh=0.10 █████
  realization          F1=0.284 thresh=0.70 █████
  confusion            F1=0.322 thresh=0.75 ██████
  excitement           F1=0.346 thresh=0.10 ██████
  desire               F1=0.352 thresh=0.25 ███████
  disappointment       F1=0.376 thresh=0.60 ███████
  admiration           F1=0.392 thresh=0.75 ███████
  neutral              F1=0.397 thresh=0.65 ███████
  curiosity            F1=0.430 thresh=0.15 ████████
  annoyance            F1=0.457 thresh=0.45 █████████

✅ Nhãn F1 cao nhất (top 10):
  caring               F1=0.547 thresh=0.50 ██████████
  love                 F1=0.554 thresh=0.75 ███████████
  joy                  F1=0.599 thresh=0.80 ███████████
  pride                F1=0.619 thresh=0.85 ████████████
  optimism             F1=0.646 thresh=0.60 ████████████
  remorse              F1=0.688 thresh=0.90 █████████████
  fear                 F1=0.718 thresh=0.90 ██████████████
  sadness              F1=0.724 thresh=0.85 ██████████████
  gratitude            F1=0.820 thresh=0.85 ████████████████
  embarrassment        F1=0.833 thresh=0.85 ████████████████

==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.86      0.86      0.86       599
   Offensive       0.72      0.72      0.72       496
        Hate       0.78      0.78      0.78       380

    accuracy                           0.79      1475
   macro avg       0.79      0.79      0.79      1475
weighted avg       0.79      0.79      0.79      1475

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .