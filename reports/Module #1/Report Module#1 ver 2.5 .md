✅ Project Root: /kaggle/input/datasets/ptg11082005/mental-health/project_backup
🚀 Device: cuda
   Ver 2.5: No Sampler (shuffle=True) + preprocess fix
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Emotion  train=22022 | valid=2742
✅ Hate     train=5163 | valid=738

📊 Train total: 27185 | Val total: 3480
   Loader: shuffle=True (NO WeightedRandomSampler)
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  | 
--------------------------------+------------+--+-
lm_head.dense.weight            | UNEXPECTED |  | 
roberta.embeddings.position_ids | UNEXPECTED |  | 
lm_head.bias                    | UNEXPECTED |  | 
lm_head.layer_norm.bias         | UNEXPECTED |  | 
lm_head.layer_norm.weight       | UNEXPECTED |  | 
lm_head.decoder.bias            | UNEXPECTED |  | 
lm_head.dense.bias              | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
   pos_weight — min=4.9 | max=20.0 | mean=16.6

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/30: 100%|██████████| 2266/2266 [10:39<00:00,  3.54it/s, E=0.96, H=0.47, loss=0.7143]
📊 Epoch 1: Loss=1.0699 | NaN batches skipped=0
   Emotion → F1 micro=0.1793 | F1 macro=0.0807
   Hate    → Acc=0.6328
   Combined score: 0.3607
⭐ Đã lưu best model (score=0.3607)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/30: 100%|██████████| 2266/2266 [11:03<00:00,  3.41it/s, E=0.62, H=0.91, loss=0.7632]
📊 Epoch 2: Loss=0.9061 | NaN batches skipped=0
   Emotion → F1 micro=0.2968 | F1 macro=0.2839
   Hate    → Acc=0.7249
   Combined score: 0.4680
⭐ Đã lưu best model (score=0.4680)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.74, H=0.00, loss=0.3697]
📊 Epoch 3: Loss=0.7213 | NaN batches skipped=0
   Emotion → F1 micro=0.3549 | F1 macro=0.3480
   Hate    → Acc=0.7331
   Combined score: 0.5061
⭐ Đã lưu best model (score=0.5061)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.33, H=0.00, loss=0.2313]
📊 Epoch 4: Loss=0.6326 | NaN batches skipped=0
   Emotion → F1 micro=0.3994 | F1 macro=0.4063
   Hate    → Acc=0.8008
   Combined score: 0.5600
⭐ Đã lưu best model (score=0.5600)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.42, H=0.30, loss=0.3801]
📊 Epoch 5: Loss=0.5332 | NaN batches skipped=0
   Emotion → F1 micro=0.4340 | F1 macro=0.4325
   Hate    → Acc=0.7913
   Combined score: 0.5770
⭐ Đã lưu best model (score=0.5770)

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.95, H=0.34, loss=0.7652]
📊 Epoch 6: Loss=0.4663 | NaN batches skipped=0
   Emotion → F1 micro=0.4522 | F1 macro=0.4599
   Hate    → Acc=0.7995
   Combined score: 0.5911
⭐ Đã lưu best model (score=0.5911)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/30: 100%|██████████| 2266/2266 [11:02<00:00,  3.42it/s, E=0.51, H=0.00, loss=0.3539]
📊 Epoch 7: Loss=0.4131 | NaN batches skipped=0
   Emotion → F1 micro=0.4592 | F1 macro=0.4596
   Hate    → Acc=0.8022
   Combined score: 0.5964
⭐ Đã lưu best model (score=0.5964)

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/30: 100%|██████████| 2266/2266 [10:52<00:00,  3.47it/s, E=0.27, H=0.00, loss=0.1923]
📊 Epoch 8: Loss=0.3762 | NaN batches skipped=0
   Emotion → F1 micro=0.4671 | F1 macro=0.4727
   Hate    → Acc=0.7967
   Combined score: 0.5990
⭐ Đã lưu best model (score=0.5990)

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.26, H=0.00, loss=0.1812]
📊 Epoch 9: Loss=0.3441 | NaN batches skipped=0
   Emotion → F1 micro=0.4831 | F1 macro=0.4931
   Hate    → Acc=0.7900
   Combined score: 0.6059
⭐ Đã lưu best model (score=0.6059)

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.28, H=0.30, loss=0.2834]
📊 Epoch 10: Loss=0.3201 | NaN batches skipped=0
   Emotion → F1 micro=0.4965 | F1 macro=0.4939
   Hate    → Acc=0.8049
   Combined score: 0.6198
⭐ Đã lưu best model (score=0.6198)

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.54, H=0.32, loss=0.4730]
📊 Epoch 11: Loss=0.3028 | NaN batches skipped=0
   Emotion → F1 micro=0.4903 | F1 macro=0.4873
   Hate    → Acc=0.7818
   Combined score: 0.6069

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/30: 100%|██████████| 2266/2266 [10:57<00:00,  3.45it/s, E=0.37, H=0.00, loss=0.2564]
📊 Epoch 12: Loss=0.2819 | NaN batches skipped=0
   Emotion → F1 micro=0.4962 | F1 macro=0.5026
   Hate    → Acc=0.8008
   Combined score: 0.6180

--- Chiến lược Epoch 13: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 13/30: 100%|██████████| 2266/2266 [10:56<00:00,  3.45it/s, E=0.19, H=0.29, loss=0.2189]
📊 Epoch 13: Loss=0.2691 | NaN batches skipped=0
   Emotion → F1 micro=0.5002 | F1 macro=0.5032
   Hate    → Acc=0.8144
   Combined score: 0.6259
⭐ Đã lưu best model (score=0.6259)

--- Chiến lược Epoch 14: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 14/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.20, H=0.00, loss=0.1389]
📊 Epoch 14: Loss=0.2567 | NaN batches skipped=0
   Emotion → F1 micro=0.5109 | F1 macro=0.5097
   Hate    → Acc=0.7995
   Combined score: 0.6263
⭐ Đã lưu best model (score=0.6263)

--- Chiến lược Epoch 15: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 15/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.26, H=0.00, loss=0.1846]
📊 Epoch 15: Loss=0.2445 | NaN batches skipped=0
   Emotion → F1 micro=0.5055 | F1 macro=0.5058
   Hate    → Acc=0.7995
   Combined score: 0.6231

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.18, H=0.29, loss=0.2055]
📊 Epoch 16: Loss=0.2268 | NaN batches skipped=0
   Emotion → F1 micro=0.5090 | F1 macro=0.5051
   Hate    → Acc=0.7940
   Combined score: 0.6230

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/30: 100%|██████████| 2266/2266 [10:53<00:00,  3.47it/s, E=0.19, H=0.00, loss=0.1484]
📊 Epoch 17: Loss=0.2182 | NaN batches skipped=0
   Emotion → F1 micro=0.5128 | F1 macro=0.5145
   Hate    → Acc=0.7981
   Combined score: 0.6269
⭐ Đã lưu best model (score=0.6269)

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/30: 100%|██████████| 2266/2266 [11:02<00:00,  3.42it/s, E=0.17, H=0.00, loss=0.1357]
📊 Epoch 18: Loss=0.2072 | NaN batches skipped=0
   Emotion → F1 micro=0.5128 | F1 macro=0.5127
   Hate    → Acc=0.8035
   Combined score: 0.6291
⭐ Đã lưu best model (score=0.6291)

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.13, H=0.30, loss=0.1614]
📊 Epoch 19: Loss=0.1990 | NaN batches skipped=0
   Emotion → F1 micro=0.5068 | F1 macro=0.5035
   Hate    → Acc=0.8022
   Combined score: 0.6249

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.24, H=0.00, loss=0.1926]
📊 Epoch 20: Loss=0.1900 | NaN batches skipped=0
   Emotion → F1 micro=0.5074 | F1 macro=0.5012
   Hate    → Acc=0.8049
   Combined score: 0.6264

--- Chiến lược Epoch 21: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 21/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.14, H=0.00, loss=0.1082]
📊 Epoch 21: Loss=0.1845 | NaN batches skipped=0
   Emotion → F1 micro=0.5051 | F1 macro=0.5036
   Hate    → Acc=0.7913
   Combined score: 0.6196

--- Chiến lược Epoch 22: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 22/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.18, H=0.00, loss=0.1462]
📊 Epoch 22: Loss=0.1774 | NaN batches skipped=0
   Emotion → F1 micro=0.5046 | F1 macro=0.5013
   Hate    → Acc=0.7900
   Combined score: 0.6188

--- Chiến lược Epoch 23: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 23/30: 100%|██████████| 2266/2266 [11:02<00:00,  3.42it/s, E=0.18, H=0.00, loss=0.1409]
📊 Epoch 23: Loss=0.1729 | NaN batches skipped=0
   Emotion → F1 micro=0.5042 | F1 macro=0.5015
   Hate    → Acc=0.7927
   Combined score: 0.6196

--- Chiến lược Epoch 24: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 24/30: 100%|██████████| 2266/2266 [10:51<00:00,  3.48it/s, E=0.07, H=0.30, loss=0.1152]
📊 Epoch 24: Loss=0.1676 | NaN batches skipped=0
   Emotion → F1 micro=0.5053 | F1 macro=0.5005
   Hate    → Acc=0.7940
   Combined score: 0.6208

--- Chiến lược Epoch 25: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 25/30: 100%|██████████| 2266/2266 [10:50<00:00,  3.48it/s, E=0.03, H=0.33, loss=0.0886]
📊 Epoch 25: Loss=0.1654 | NaN batches skipped=0
   Emotion → F1 micro=0.5011 | F1 macro=0.5005
   Hate    → Acc=0.7940
   Combined score: 0.6183
🛑 Early stopping tại Epoch 25

✅ Training hoàn tất.
   Best combined score: 0.6291
   Tổng NaN batches skipped: 0
   

Dang kich hoat moi truong ao Py.12...
Dang kiem tra GPU...
CUDA Ready: True
Bat dau kiem tra mo hinh PhoBERT...
📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Đánh giá trên: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ Loaded tuned thresholds từ: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\checkpoints\Module#1 ver 2.5\emotion_thresholds.json
✅ Emotion test: 2751 | Hate test: 1475
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8393.50it/s]
RobertaModel LOAD REPORT from: vinai/phobert-base
Key                             | Status     |  |
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  |
lm_head.dense.weight            | UNEXPECTED |  |
lm_head.decoder.bias            | UNEXPECTED |  |
lm_head.bias                    | UNEXPECTED |  |
lm_head.layer_norm.weight       | UNEXPECTED |  |
lm_head.layer_norm.bias         | UNEXPECTED |  |
lm_head.dense.bias              | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📊 Đang dự đoán...

==================== EMOTION REPORT (Multi-label 28 nhãn) ====================

📊 So sánh threshold:
   @threshold=0.5:    F1 micro=0.5230 | F1 macro=0.5156
   @tuned threshold:  F1 micro=0.5244 | F1 macro=0.5210  (+0.0013 micro | +0.0053 macro)

⚠️  Nhãn F1 thấp nhất (bottom 10):
  disapproval          F1=0.267 thresh=0.35 █████
  confusion            F1=0.339 thresh=0.65 ██████
  realization          F1=0.350 thresh=0.80 ██████
  disappointment       F1=0.361 thresh=0.75 ███████
  desire               F1=0.379 thresh=0.55 ███████
  neutral              F1=0.399 thresh=0.35 ███████
  excitement           F1=0.406 thresh=0.20 ████████
  admiration           F1=0.415 thresh=0.50 ████████
  curiosity            F1=0.433 thresh=0.55 ████████
  nervousness          F1=0.445 thresh=0.45 ████████

✅ Nhãn F1 cao nhất (top 10):
  anger                F1=0.550 thresh=0.65 ███████████
  caring               F1=0.555 thresh=0.75 ███████████
  pride                F1=0.610 thresh=0.90 ████████████
  joy                  F1=0.618 thresh=0.50 ████████████
  optimism             F1=0.635 thresh=0.70 ████████████
  fear                 F1=0.718 thresh=0.60 ██████████████
  sadness              F1=0.731 thresh=0.85 ██████████████
  remorse              F1=0.732 thresh=0.35 ██████████████
  embarrassment        F1=0.793 thresh=0.45 ███████████████
  gratitude            F1=0.827 thresh=0.85 ████████████████

==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.88      0.86      0.87       599
   Offensive       0.72      0.78      0.75       496
        Hate       0.82      0.76      0.79       380

    accuracy                           0.81      1475
   macro avg       0.80      0.80      0.80      1475
weighted avg       0.81      0.81      0.81      1475

Da hoan thanh! Nhan phim bat ky de thoat.
Press any key to continue . . .