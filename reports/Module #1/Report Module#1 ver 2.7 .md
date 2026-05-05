✅ Project Root: /kaggle/input/datasets/ptg11082005/mental-health-ver2-7
📁 project_root: /kaggle/input/datasets/ptg11082005/mental-health-ver2-7
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
🚀 Device: cuda
   Ver 2.7: MAX_LEN=192 | BATCH=8 | Focal γ=1.0 | Smooth ε=0.05 | Freeze 2 epochs
✅ Emotion  train=21001 | valid=2601
✅ Hate     train=5163 | valid=738

📊 Train total: 26164 | Val total: 3339
RobertaModel LOAD REPORT from: vinai/phobert-base-v2
Key                             | Status     | 
--------------------------------+------------+-
lm_head.bias                    | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.dense.weight            | UNEXPECTED | 
lm_head.dense.bias              | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
pooler.dense.bias               | MISSING    | 
pooler.dense.weight             | MISSING    | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
   pos_weight — min=4.6 | max=15.0 | mean=13.2
🔒 PhoBERT frozen (epochs 1-2)

--- Chiến lược Epoch 1: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 1/30: 100%|██████████| 3271/3271 [05:07<00:00, 10.63it/s, E=0.90, H=0.00, loss=0.4497]
📊 Epoch 1: Loss=0.8006 | NaN batches skipped=0
   Emotion → F1 micro=0.2188 | F1 macro=0.1229
   Hate    → Acc=0.6396
   Combined score: 0.3871
⭐ Đã lưu best model (score=0.3871)

--- Chiến lược Epoch 2: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 2/30: 100%|██████████| 3271/3271 [05:42<00:00,  9.56it/s, E=0.68, H=0.00, loss=0.3419]
📊 Epoch 2: Loss=0.7002 | NaN batches skipped=0
   Emotion → F1 micro=0.2834 | F1 macro=0.2580
   Hate    → Acc=0.6734
   Combined score: 0.4394
⭐ Đã lưu best model (score=0.4394)
🔓 PhoBERT unfrozen (epoch 3+)

--- Chiến lược Epoch 3: Warm-up: Equal Focus [E:0.5 - H:0.5] ---
✨ Epoch 3/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.18it/s, E=0.47, H=0.42, loss=0.4478]
📊 Epoch 3: Loss=0.6346 | NaN batches skipped=0
   Emotion → F1 micro=0.3668 | F1 macro=0.3718
   Hate    → Acc=0.7602
   Combined score: 0.5241
⭐ Đã lưu best model (score=0.5241)

--- Chiến lược Epoch 4: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 4/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.61, H=0.55, loss=0.5924]
📊 Epoch 4: Loss=0.5239 | NaN batches skipped=0
   Emotion → F1 micro=0.4468 | F1 macro=0.4406
   Hate    → Acc=0.7954
   Combined score: 0.5862
⭐ Đã lưu best model (score=0.5862)

--- Chiến lược Epoch 5: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 5/30: 100%|██████████| 3271/3271 [17:09<00:00,  3.18it/s, E=0.36, H=0.00, loss=0.2542]
📊 Epoch 5: Loss=0.4570 | NaN batches skipped=0
   Emotion → F1 micro=0.4467 | F1 macro=0.4424
   Hate    → Acc=0.7805
   Combined score: 0.5802

--- Chiến lược Epoch 6: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 6/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.31, H=0.00, loss=0.2141]
📊 Epoch 6: Loss=0.4058 | NaN batches skipped=0
   Emotion → F1 micro=0.4780 | F1 macro=0.4958
   Hate    → Acc=0.7927
   Combined score: 0.6039
⭐ Đã lưu best model (score=0.6039)

--- Chiến lược Epoch 7: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 7/30: 100%|██████████| 3271/3271 [17:11<00:00,  3.17it/s, E=0.35, H=0.30, loss=0.3390]
📊 Epoch 7: Loss=0.3700 | NaN batches skipped=0
   Emotion → F1 micro=0.4895 | F1 macro=0.4934
   Hate    → Acc=0.8049
   Combined score: 0.6156
⭐ Đã lưu best model (score=0.6156)

--- Chiến lược Epoch 8: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 8/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.30, H=0.29, loss=0.2952]
📊 Epoch 8: Loss=0.3457 | NaN batches skipped=0
   Emotion → F1 micro=0.5099 | F1 macro=0.5071
   Hate    → Acc=0.8008
   Combined score: 0.6263
⭐ Đã lưu best model (score=0.6263)

--- Chiến lược Epoch 9: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 9/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.32, H=0.30, loss=0.3153]
📊 Epoch 9: Loss=0.3278 | NaN batches skipped=0
   Emotion → F1 micro=0.5133 | F1 macro=0.5203
   Hate    → Acc=0.8117
   Combined score: 0.6327
⭐ Đã lưu best model (score=0.6327)

--- Chiến lược Epoch 10: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 10/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.25, H=0.00, loss=0.1771]
📊 Epoch 10: Loss=0.3112 | NaN batches skipped=0
   Emotion → F1 micro=0.5214 | F1 macro=0.5276
   Hate    → Acc=0.8089
   Combined score: 0.6364
⭐ Đã lưu best model (score=0.6364)

--- Chiến lược Epoch 11: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 11/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.23, H=0.00, loss=0.1624]
📊 Epoch 11: Loss=0.3014 | NaN batches skipped=0
   Emotion → F1 micro=0.5289 | F1 macro=0.5334
   Hate    → Acc=0.8049
   Combined score: 0.6393
⭐ Đã lưu best model (score=0.6393)

--- Chiến lược Epoch 12: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 12/30: 100%|██████████| 3271/3271 [17:09<00:00,  3.18it/s, E=0.56, H=0.00, loss=0.3928]
📊 Epoch 12: Loss=0.2907 | NaN batches skipped=0
   Emotion → F1 micro=0.5350 | F1 macro=0.5308
   Hate    → Acc=0.8198
   Combined score: 0.6489
⭐ Đã lưu best model (score=0.6489)

--- Chiến lược Epoch 13: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 13/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.26, H=0.29, loss=0.2698]
📊 Epoch 13: Loss=0.2815 | NaN batches skipped=0
   Emotion → F1 micro=0.5357 | F1 macro=0.5344
   Hate    → Acc=0.8049
   Combined score: 0.6434

--- Chiến lược Epoch 14: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 14/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.28, H=0.00, loss=0.1992]
📊 Epoch 14: Loss=0.2753 | NaN batches skipped=0
   Emotion → F1 micro=0.5408 | F1 macro=0.5363
   Hate    → Acc=0.8089
   Combined score: 0.6481

--- Chiến lược Epoch 15: Main Training: Emotion Focus [E:0.7 - H:0.3] ---
✨ Epoch 15/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.21, H=0.30, loss=0.2329]
📊 Epoch 15: Loss=0.2698 | NaN batches skipped=0
   Emotion → F1 micro=0.5407 | F1 macro=0.5320
   Hate    → Acc=0.8184
   Combined score: 0.6518
⭐ Đã lưu best model (score=0.6518)

--- Chiến lược Epoch 16: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 16/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.18it/s, E=0.30, H=0.29, loss=0.2982]
📊 Epoch 16: Loss=0.2663 | NaN batches skipped=0
   Emotion → F1 micro=0.5405 | F1 macro=0.5346
   Hate    → Acc=0.8144
   Combined score: 0.6501

--- Chiến lược Epoch 17: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 17/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.18it/s, E=0.25, H=0.30, loss=0.2570]
📊 Epoch 17: Loss=0.2605 | NaN batches skipped=0
   Emotion → F1 micro=0.5410 | F1 macro=0.5359
   Hate    → Acc=0.8279
   Combined score: 0.6558
⭐ Đã lưu best model (score=0.6558)

--- Chiến lược Epoch 18: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 18/30: 100%|██████████| 3271/3271 [17:11<00:00,  3.17it/s, E=0.22, H=0.30, loss=0.2361]
📊 Epoch 18: Loss=0.2557 | NaN batches skipped=0
   Emotion → F1 micro=0.5390 | F1 macro=0.5339
   Hate    → Acc=0.8157
   Combined score: 0.6497

--- Chiến lược Epoch 19: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 19/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.23, H=0.00, loss=0.1878]
📊 Epoch 19: Loss=0.2513 | NaN batches skipped=0
   Emotion → F1 micro=0.5370 | F1 macro=0.5345
   Hate    → Acc=0.8062
   Combined score: 0.6447

--- Chiến lược Epoch 20: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 20/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.26, H=0.29, loss=0.2640]
📊 Epoch 20: Loss=0.2482 | NaN batches skipped=0
   Emotion → F1 micro=0.5366 | F1 macro=0.5341
   Hate    → Acc=0.8225
   Combined score: 0.6510

--- Chiến lược Epoch 21: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 21/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.37, H=0.00, loss=0.2943]
📊 Epoch 21: Loss=0.2448 | NaN batches skipped=0
   Emotion → F1 micro=0.5395 | F1 macro=0.5361
   Hate    → Acc=0.8198
   Combined score: 0.6516

--- Chiến lược Epoch 22: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 22/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.18, H=0.30, loss=0.2068]
📊 Epoch 22: Loss=0.2412 | NaN batches skipped=0
   Emotion → F1 micro=0.5339 | F1 macro=0.5305
   Hate    → Acc=0.8171
   Combined score: 0.6472

--- Chiến lược Epoch 23: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 23/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.19, H=0.29, loss=0.2087]
📊 Epoch 23: Loss=0.2392 | NaN batches skipped=0
   Emotion → F1 micro=0.5322 | F1 macro=0.5302
   Hate    → Acc=0.8198
   Combined score: 0.6473

--- Chiến lược Epoch 24: Final Squeeze: Max Emotion Focus [E:0.8 - H:0.2] ---
✨ Epoch 24/30: 100%|██████████| 3271/3271 [17:10<00:00,  3.17it/s, E=0.19, H=0.00, loss=0.1526]
📊 Epoch 24: Loss=0.2359 | NaN batches skipped=0
   Emotion → F1 micro=0.5291 | F1 macro=0.5273
   Hate    → Acc=0.8252
   Combined score: 0.6476
🛑 Early stopping tại Epoch 24

✅ Training hoàn tất.
   Best combined score: 0.6558
   Tổng NaN batches skipped: 0





