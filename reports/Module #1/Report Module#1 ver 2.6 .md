✅ Project Root: /kaggle/input/datasets/ptg11082005/train-test-module-ver2-6/project_backup
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
🚀 Device: cuda
   Ver 2.5: No Sampler (shuffle=True) + preprocess fix
✅ Emotion  train=21001 | valid=2601
✅ Hate     train=5163 | valid=738

📊 Train total: 26164 | Val total: 3339
   Loader: shuffle=True (NO WeightedRandomSampler)
RobertaModel LOAD REPORT from: vinai/phobert-base-v2
Key                             | Status     | 
--------------------------------+------------+-
lm_head.bias                    | UNEXPECTED | 
lm_head.dense.bias              | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.dense.weight            | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
pooler.dense.bias               | MISSING    | 
pooler.dense.weight             | MISSING    | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
   pos_weight — min=4.6 | max=20.0 | mean=16.7

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/30: 100%|██████████| 2181/2181 [10:16<00:00,  3.54it/s, E=1.27, H=1.01, loss=1.1369]
📊 Epoch 1: Loss=1.0514 | NaN batches skipped=0
   Emotion → F1 micro=0.2342 | F1 macro=0.1309
   Hate    → Acc=0.6938
   Combined score: 0.4180
⭐ Đã lưu best model (score=0.4180)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.76, H=0.00, loss=0.3821]
📊 Epoch 2: Loss=0.8623 | NaN batches skipped=0
   Emotion → F1 micro=0.3135 | F1 macro=0.2928
   Hate    → Acc=0.7412
   Combined score: 0.4846
⭐ Đã lưu best model (score=0.4846)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.55, H=0.00, loss=0.2753]
📊 Epoch 3: Loss=0.6952 | NaN batches skipped=0
   Emotion → F1 micro=0.3858 | F1 macro=0.3709
   Hate    → Acc=0.7791
   Combined score: 0.5431
⭐ Đã lưu best model (score=0.5431)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=1.06, H=0.33, loss=0.8394]
📊 Epoch 4: Loss=0.6153 | NaN batches skipped=0
   Emotion → F1 micro=0.4131 | F1 macro=0.4025
   Hate    → Acc=0.7710
   Combined score: 0.5563
⭐ Đã lưu best model (score=0.5563)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=1.44, H=0.00, loss=1.0100]
📊 Epoch 5: Loss=0.5183 | NaN batches skipped=0
   Emotion → F1 micro=0.4693 | F1 macro=0.4687
   Hate    → Acc=0.7737
   Combined score: 0.5911
⭐ Đã lưu best model (score=0.5911)

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.59, H=0.39, loss=0.5325]
📊 Epoch 6: Loss=0.4514 | NaN batches skipped=0
   Emotion → F1 micro=0.4674 | F1 macro=0.4648
   Hate    → Acc=0.8184
   Combined score: 0.6078
⭐ Đã lưu best model (score=0.6078)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.26, H=0.57, loss=0.3557]
📊 Epoch 7: Loss=0.4016 | NaN batches skipped=0
   Emotion → F1 micro=0.4898 | F1 macro=0.4864
   Hate    → Acc=0.8157
   Combined score: 0.6202
⭐ Đã lưu best model (score=0.6202)

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=1.15, H=0.29, loss=0.8944]
📊 Epoch 8: Loss=0.3636 | NaN batches skipped=0
   Emotion → F1 micro=0.4919 | F1 macro=0.4839
   Hate    → Acc=0.8320
   Combined score: 0.6280
⭐ Đã lưu best model (score=0.6280)

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.34, H=0.00, loss=0.2375]
📊 Epoch 9: Loss=0.3304 | NaN batches skipped=0
   Emotion → F1 micro=0.5025 | F1 macro=0.4990
   Hate    → Acc=0.8089
   Combined score: 0.6251

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.25, H=0.00, loss=0.1759]
📊 Epoch 10: Loss=0.3077 | NaN batches skipped=0
   Emotion → F1 micro=0.5127 | F1 macro=0.5103
   Hate    → Acc=0.8225
   Combined score: 0.6366
⭐ Đã lưu best model (score=0.6366)

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.19, H=0.00, loss=0.1353]
📊 Epoch 11: Loss=0.2883 | NaN batches skipped=0
   Emotion → F1 micro=0.5162 | F1 macro=0.5177
   Hate    → Acc=0.8008
   Combined score: 0.6301

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.34, H=0.30, loss=0.3300]
📊 Epoch 12: Loss=0.2729 | NaN batches skipped=0
   Emotion → F1 micro=0.5240 | F1 macro=0.5193
   Hate    → Acc=0.8130
   Combined score: 0.6396
⭐ Đã lưu best model (score=0.6396)

--- Chiến lược Epoch 13: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 13/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.23, H=0.00, loss=0.1623]
📊 Epoch 13: Loss=0.2571 | NaN batches skipped=0
   Emotion → F1 micro=0.5322 | F1 macro=0.5258
   Hate    → Acc=0.8266
   Combined score: 0.6499
⭐ Đã lưu best model (score=0.6499)

--- Chiến lược Epoch 14: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 14/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.22, H=0.00, loss=0.1521]
📊 Epoch 14: Loss=0.2433 | NaN batches skipped=0
   Emotion → F1 micro=0.5298 | F1 macro=0.5259
   Hate    → Acc=0.8266
   Combined score: 0.6485

--- Chiến lược Epoch 15: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 15/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.93, H=0.30, loss=0.7432]
📊 Epoch 15: Loss=0.2336 | NaN batches skipped=0
   Emotion → F1 micro=0.5239 | F1 macro=0.5126
   Hate    → Acc=0.8144
   Combined score: 0.6401

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.25, H=0.30, loss=0.2628]
📊 Epoch 16: Loss=0.2153 | NaN batches skipped=0
   Emotion → F1 micro=0.5385 | F1 macro=0.5311
   Hate    → Acc=0.8252
   Combined score: 0.6532
⭐ Đã lưu best model (score=0.6532)

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.20, H=0.35, loss=0.2266]
📊 Epoch 17: Loss=0.2052 | NaN batches skipped=0
   Emotion → F1 micro=0.5345 | F1 macro=0.5307
   Hate    → Acc=0.8225
   Combined score: 0.6497

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.17, H=0.31, loss=0.1997]
📊 Epoch 18: Loss=0.1945 | NaN batches skipped=0
   Emotion → F1 micro=0.5346 | F1 macro=0.5289
   Hate    → Acc=0.8252
   Combined score: 0.6508

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/30: 100%|██████████| 2181/2181 [10:18<00:00,  3.53it/s, E=0.09, H=0.29, loss=0.1322]
📊 Epoch 19: Loss=0.1854 | NaN batches skipped=0
   Emotion → F1 micro=0.5354 | F1 macro=0.5287
   Hate    → Acc=0.8171
   Combined score: 0.6481

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.06, H=0.00, loss=0.0511]
📊 Epoch 20: Loss=0.1784 | NaN batches skipped=0
   Emotion → F1 micro=0.5300 | F1 macro=0.5221
   Hate    → Acc=0.8211
   Combined score: 0.6464

--- Chiến lược Epoch 21: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 21/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.11, H=0.29, loss=0.1502]
📊 Epoch 21: Loss=0.1731 | NaN batches skipped=0
   Emotion → F1 micro=0.5300 | F1 macro=0.5257
   Hate    → Acc=0.8198
   Combined score: 0.6459

--- Chiến lược Epoch 22: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 22/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.16, H=0.00, loss=0.1292]
📊 Epoch 22: Loss=0.1660 | NaN batches skipped=0
   Emotion → F1 micro=0.5311 | F1 macro=0.5260
   Hate    → Acc=0.8184
   Combined score: 0.6460

--- Chiến lược Epoch 23: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 23/30: 100%|██████████| 2181/2181 [10:17<00:00,  3.53it/s, E=0.06, H=0.00, loss=0.0442]
📊 Epoch 23: Loss=0.1607 | NaN batches skipped=0
   Emotion → F1 micro=0.5247 | F1 macro=0.5210
   Hate    → Acc=0.8144
   Combined score: 0.6406
🛑 Early stopping tại Epoch 23

✅ Training hoàn tất.
   Best combined score: 0.6532
   Tổng NaN batches skipped: 0

Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: False
Bat dau kiem tra mo hinh PhoBERT...
📁 project_root: C:\Users\i3admin\Downloads\Giangf\MentalHealth_AI_Project
🚀 Đánh giá trên: cpu
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Loaded tuned thresholds từ: C:\Users\i3admin\Downloads\Giangf\MentalHealth_AI_Project\checkpoints\Module#1 ver 2.6\emotion_thresholds.json
✅ Emotion test: 2622 | Hate test: 1475
Loading weights: 100%|█████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 7005.20it/s]
[transformers] RobertaModel LOAD REPORT from: vinai/phobert-base-v2
Key                       | Status     |
--------------------------+------------+-
lm_head.layer_norm.weight | UNEXPECTED |
lm_head.layer_norm.bias   | UNEXPECTED |
lm_head.dense.weight      | UNEXPECTED |
lm_head.dense.bias        | UNEXPECTED |
lm_head.bias              | UNEXPECTED |
pooler.dense.bias         | MISSING    |
pooler.dense.weight       | MISSING    |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

📊 Đang dự đoán...

==================== EMOTION REPORT (Multi-label 28 nhãn) ====================

📊 So sánh threshold:
   @threshold=0.5:    F1 micro=0.5431 | F1 macro=0.5285
   @tuned threshold:  F1 micro=0.5420 | F1 macro=0.5402  (+-0.0012 micro | +0.0117 macro)

⚠️  Nhãn F1 thấp nhất (bottom 10):
  disapproval          F1=0.282 thresh=0.20 █████
  realization          F1=0.344 thresh=0.50 ██████
  confusion            F1=0.368 thresh=0.90 ███████
  excitement           F1=0.374 thresh=0.20 ███████
  desire               F1=0.392 thresh=0.40 ███████
  neutral              F1=0.393 thresh=0.15 ███████
  nervousness          F1=0.407 thresh=0.45 ████████
  disappointment       F1=0.413 thresh=0.45 ████████
  admiration           F1=0.431 thresh=0.55 ████████
  annoyance            F1=0.504 thresh=0.50 ██████████

✅ Nhãn F1 cao nhất (top 10):
  curiosity            F1=0.581 thresh=0.85 ███████████
  caring               F1=0.598 thresh=0.60 ███████████
  joy                  F1=0.603 thresh=0.60 ████████████
  pride                F1=0.651 thresh=0.75 █████████████
  optimism             F1=0.674 thresh=0.80 █████████████
  remorse              F1=0.698 thresh=0.85 █████████████
  sadness              F1=0.733 thresh=0.85 ██████████████
  fear                 F1=0.742 thresh=0.70 ██████████████
  gratitude            F1=0.810 thresh=0.90 ████████████████
  embarrassment        F1=0.812 thresh=0.85 ████████████████

==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.88      0.86      0.87       599
   Offensive       0.73      0.77      0.75       496
        Hate       0.82      0.79      0.80       380

    accuracy                           0.81      1475
   macro avg       0.81      0.81      0.81      1475
weighted avg       0.81      0.81      0.81      1475

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .