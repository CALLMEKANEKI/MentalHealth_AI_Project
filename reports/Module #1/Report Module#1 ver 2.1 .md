# Module#1 Ver 2.1
# Đổi lại cấu trúc model từ 7 -> 28 nhãn
# Trả về tương ứng 28 chiều 

# ----------------Training-----------------------#
# Best model: Epoch 20
# Emotion → F1 micro=0.4762 | F1 macro=0.4649
# Hate    → Acc=0.8022
# Combined score: 0.6066

📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Device: cuda
✅ Emotion  train=22022 | valid=2742
✅ Hate     train=5163 | valid=738

📊 Train total: 27185 | Val total: 3480
Loading weights: 100%|█████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3581.80it/s]

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/20: 100%|██████████████████████████████████████| 2266/2266 [20:46<00:00,  1.82it/s, E=0.22, H=nan, loss=nan]
📊 Epoch 1: Loss=nan
   Emotion → F1 micro=0.0000 | F1 macro=0.0000
   Hate    → Acc=0.6450
   Combined score: 0.2580
⭐ Đã lưu best model (score=0.2580)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/20: 100%|██████████████████████████████████| 2266/2266 [20:43<00:00,  1.82it/s, E=0.17, H=0.71, loss=0.4367]
📊 Epoch 2: Loss=nan
   Emotion → F1 micro=0.1138 | F1 macro=0.0204
   Hate    → Acc=0.7168
   Combined score: 0.3550
⭐ Đã lưu best model (score=0.3550)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/20: 100%|██████████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.18, H=nan, loss=nan]
📊 Epoch 3: Loss=nan
   Emotion → F1 micro=0.2179 | F1 macro=0.1215
   Hate    → Acc=0.7493
   Combined score: 0.4305
⭐ Đã lưu best model (score=0.4305)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/20: 100%|██████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.05, H=0.34, loss=0.1385]
📊 Epoch 4: Loss=nan
   Emotion → F1 micro=0.3708 | F1 macro=0.2727
   Hate    → Acc=0.7710
   Combined score: 0.5309
⭐ Đã lưu best model (score=0.5309)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/20: 100%|██████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.18, H=0.32, loss=0.2188]
📊 Epoch 5: Loss=nan
   Emotion → F1 micro=0.4109 | F1 macro=0.3275
   Hate    → Acc=0.7913
   Combined score: 0.5631
⭐ Đã lưu best model (score=0.5631)

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/20: 100%|██████████████████████████████████| 2266/2266 [20:28<00:00,  1.84it/s, E=0.19, H=1.48, loss=0.5772]
📊 Epoch 6: Loss=nan
   Emotion → F1 micro=0.4379 | F1 macro=0.3720
   Hate    → Acc=0.8035
   Combined score: 0.5841
⭐ Đã lưu best model (score=0.5841)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/20: 100%|██████████████████████████████████████| 2266/2266 [20:28<00:00,  1.84it/s, E=0.19, H=nan, loss=nan]
📊 Epoch 7: Loss=nan
   Emotion → F1 micro=0.4658 | F1 macro=0.4171
   Hate    → Acc=0.7927
   Combined score: 0.5966
⭐ Đã lưu best model (score=0.5966)

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/20: 100%|██████████████████████████████████| 2266/2266 [20:28<00:00,  1.84it/s, E=0.17, H=0.31, loss=0.2109]
📊 Epoch 8: Loss=nan
   Emotion → F1 micro=0.4727 | F1 macro=0.4321
   Hate    → Acc=0.8022
   Combined score: 0.6045
⭐ Đã lưu best model (score=0.6045)

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/20: 100%|██████████████████████████████████████| 2266/2266 [20:28<00:00,  1.84it/s, E=0.12, H=nan, loss=nan]
📊 Epoch 9: Loss=nan
   Emotion → F1 micro=0.4832 | F1 macro=0.4551
   Hate    → Acc=0.8049
   Combined score: 0.6119
⭐ Đã lưu best model (score=0.6119)

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/20: 100%|█████████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.12, H=nan, loss=nan]
📊 Epoch 10: Loss=nan
   Emotion → F1 micro=0.4855 | F1 macro=0.4619
   Hate    → Acc=0.8008
   Combined score: 0.6116

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/20: 100%|█████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.08, H=0.33, loss=0.1519]
📊 Epoch 11: Loss=nan
   Emotion → F1 micro=0.4853 | F1 macro=0.4621
   Hate    → Acc=0.8008
   Combined score: 0.6115

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/20: 100%|█████████████████████████████████| 2266/2266 [20:31<00:00,  1.84it/s, E=0.17, H=0.30, loss=0.2061]
📊 Epoch 12: Loss=nan
   Emotion → F1 micro=0.4783 | F1 macro=0.4527
   Hate    → Acc=0.8022
   Combined score: 0.6078

--- Chiến lược Epoch 13: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 13/20: 100%|█████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.07, H=0.30, loss=0.1122]
📊 Epoch 13: Loss=nan
   Emotion → F1 micro=0.4851 | F1 macro=0.4645
   Hate    → Acc=0.8022
   Combined score: 0.6119
⭐ Đã lưu best model (score=0.6119)

--- Chiến lược Epoch 14: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 14/20: 100%|█████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.02, H=0.30, loss=0.0738]
📊 Epoch 14: Loss=nan
   Emotion → F1 micro=0.4823 | F1 macro=0.4643
   Hate    → Acc=0.7981
   Combined score: 0.6086

--- Chiến lược Epoch 15: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 15/20: 100%|█████████████████████████████████████| 2266/2266 [20:28<00:00,  1.84it/s, E=0.06, H=nan, loss=nan]
📊 Epoch 15: Loss=nan
   Emotion → F1 micro=0.4817 | F1 macro=0.4686
   Hate    → Acc=0.8117
   Combined score: 0.6137
⭐ Đã lưu best model (score=0.6137)

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/20: 100%|█████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.03, H=0.29, loss=0.0838]
📊 Epoch 16: Loss=nan
   Emotion → F1 micro=0.4824 | F1 macro=0.4693
   Hate    → Acc=0.8062
   Combined score: 0.6120

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/20: 100%|█████████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.04, H=nan, loss=nan]
📊 Epoch 17: Loss=nan
   Emotion → F1 micro=0.4772 | F1 macro=0.4635
   Hate    → Acc=0.8035
   Combined score: 0.6077

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/20: 100%|█████████████████████████████████| 2266/2266 [20:29<00:00,  1.84it/s, E=0.06, H=0.29, loss=0.1030]
📊 Epoch 18: Loss=nan
   Emotion → F1 micro=0.4780 | F1 macro=0.4642
   Hate    → Acc=0.8035
   Combined score: 0.6082

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/20: 100%|█████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.05, H=0.29, loss=0.1001]
📊 Epoch 19: Loss=nan
   Emotion → F1 micro=0.4751 | F1 macro=0.4652
   Hate    → Acc=0.8089
   Combined score: 0.6087

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/20: 100%|█████████████████████████████████| 2266/2266 [20:30<00:00,  1.84it/s, E=0.04, H=0.29, loss=0.0926]
📊 Epoch 20: Loss=nan
   Emotion → F1 micro=0.4762 | F1 macro=0.4649
   Hate    → Acc=0.8022
   Combined score: 0.6066
🛑 Early stopping tại Epoch 20




# ----------------Training-----------------------#
📁 project_root: C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project
🚀 Đánh giá trên: cuda
✅ Emotion test: 2751 samples
✅ Hate test: 1475 samples
Loading weights: 100%|████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 12509.24it/s]

==================== EMOTION REPORT (Multi-label 28 nhãn) ====================
F1 micro: 0.4979 | F1 macro: 0.4780

⚠️  Nhãn F1 thấp nhất (bottom 10):
  disapproval          F1=0.181 ███
  disappointment       F1=0.204 ████
  amusement            F1=0.267 █████
  desire               F1=0.276 █████
  annoyance            F1=0.306 ██████
  excitement           F1=0.310 ██████
  realization          F1=0.314 ██████
  admiration           F1=0.315 ██████
  confusion            F1=0.327 ██████
  neutral              F1=0.364 ███████

✅ Nhãn F1 cao nhất (top 10):
  anger                F1=0.529 ██████████
  caring               F1=0.538 ██████████
  joy                  F1=0.592 ███████████
  pride                F1=0.629 ████████████
  optimism             F1=0.652 █████████████
  remorse              F1=0.692 █████████████
  sadness              F1=0.724 ██████████████
  fear                 F1=0.733 ██████████████
  gratitude            F1=0.771 ███████████████
  embarrassment        F1=0.800 ████████████████

==================== HATE SPEECH REPORT ====================
              precision    recall  f1-score   support

       Clean       0.85      0.87      0.86       599
   Offensive       0.74      0.72      0.73       496
        Hate       0.79      0.79      0.79       380

    accuracy                           0.80      1475
   macro avg       0.80      0.79      0.79      1475
weighted avg       0.80      0.80      0.80      1475

