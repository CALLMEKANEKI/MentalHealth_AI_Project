import os
import sys
import ast
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

current_dir  = os.path.dirname(os.path.abspath(__file__))
scripts_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

print(f"📁 project_root: {project_root}")

from module1.dataset import MultiTaskDataset, VIGO_EMOTIONS, NUM_VIGO_LABELS
from module1.model import PhoBERTMultiTask


def parse_labels(label_str):
    try:
        return ast.literal_eval(str(label_str))
    except Exception:
        return [27]


def find_best_thresholds():
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN    = 128
    BATCH_SIZE = 32  # val không cần nhỏ, dùng lớn cho nhanh

    model_path   = os.path.join(project_root, 'checkpoints','Module#1 ver 2.3', 'best_multitask_model.pth')
    emo_val_path = os.path.join(project_root, 'data', 'processed', 'emotion_valid.csv')
    save_path    = os.path.join(project_root, 'checkpoints','Module#1 ver 2.3', 'emotion_thresholds.json')

    print(f"🚀 Device: {DEVICE}")
    print(f"📂 Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # Load val set
    df_val = pd.read_csv(emo_val_path)
    ds_val = MultiTaskDataset(
        texts=df_val['text'].values,
        emotion_labels=df_val['labels'].apply(parse_labels).tolist(),
        hate_labels=np.full(len(df_val), -100, dtype=np.int64),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        emotion_source='vigo'
    )
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE)
    print(f"✅ Val samples: {len(ds_val)}")

    # Load model
    model = PhoBERTMultiTask(num_hate_labels=3, num_emotion_labels=NUM_VIGO_LABELS)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # ==================== COLLECT PROBS ====================
    print("\n📊 Đang collect predictions từ val set...")
    all_probs, all_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emo_logits, _  = model(input_ids, attention_mask)
            probs  = torch.sigmoid(emo_logits).cpu().numpy()
            labels = batch['emotion_labels'].numpy()
            all_probs.append(probs)
            all_true.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)  # [N, 28]
    all_true  = np.concatenate(all_true,  axis=0)  # [N, 28]
    print(f"✅ Collected {len(all_probs)} samples")

    # ==================== BASELINE @0.5 ====================
    preds_default = (all_probs > 0.5).astype(int)
    f1_micro_default = f1_score(all_true, preds_default, average='micro', zero_division=0)
    f1_macro_default = f1_score(all_true, preds_default, average='macro', zero_division=0)

    # ==================== THRESHOLD TUNING ====================
    print("\n=== THRESHOLD TUNING PER LABEL ===")
    print(f"{'Label':25s} {'Best Thresh':12s} {'F1 @0.5':10s} {'F1 @best':10s} {'Δ':8s}")
    print("-" * 70)

    best_thresholds  = []
    total_gain       = 0.0

    for i in range(NUM_VIGO_LABELS):
        true_i  = all_true[:, i]
        probs_i = all_probs[:, i]

        # F1 với threshold mặc định
        f1_default = f1_score(
            true_i, (probs_i > 0.5).astype(int), zero_division=0
        )

        # Tìm threshold tốt nhất
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds = (probs_i > thresh).astype(int)
            # Chỉ tính nếu có ít nhất 1 positive prediction
            if preds.sum() == 0:
                continue
            f1 = f1_score(true_i, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)

        best_thresholds.append(best_thresh)
        delta      = best_f1 - f1_default
        total_gain += delta

        label_name = VIGO_EMOTIONS[i] if i < len(VIGO_EMOTIONS) else f'label_{i}'
        delta_str  = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(
            f"{label_name:25s} {best_thresh:.2f}         "
            f"{f1_default:.3f}      {best_f1:.3f}      {delta_str}"
        )

    # ==================== TÍNH F1 SAU TUNING ====================
    preds_tuned = np.zeros_like(all_probs, dtype=int)
    for i, thresh in enumerate(best_thresholds):
        preds_tuned[:, i] = (all_probs[:, i] > thresh).astype(int)

    f1_micro_tuned = f1_score(all_true, preds_tuned, average='micro', zero_division=0)
    f1_macro_tuned = f1_score(all_true, preds_tuned, average='macro', zero_division=0)

    print(f"\n{'='*70}")
    print(f"📊 KẾT QUẢ TRÊN VAL SET:")
    print(f"   F1 micro @threshold=0.5:    {f1_micro_default:.4f}")
    print(f"   F1 micro @tuned threshold:  {f1_micro_tuned:.4f}  (+{f1_micro_tuned-f1_micro_default:.4f})")
    print(f"   F1 macro @threshold=0.5:    {f1_macro_default:.4f}")
    print(f"   F1 macro @tuned threshold:  {f1_macro_tuned:.4f}  (+{f1_macro_tuned-f1_macro_default:.4f})")
    print(f"   Tổng gain trung bình/nhãn:  {total_gain/NUM_VIGO_LABELS:.4f}")

    # ==================== LƯU THRESHOLDS ====================
    threshold_dict = {
        VIGO_EMOTIONS[i]: best_thresholds[i]
        for i in range(NUM_VIGO_LABELS)
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu thresholds tại: {save_path}")
    print("   Dùng file này trong inference để predict với threshold tối ưu.")

    return best_thresholds, threshold_dict


if __name__ == "__main__":
    find_best_thresholds()