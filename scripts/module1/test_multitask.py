import os
import sys
import ast
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

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


def load_thresholds(threshold_path, default=0.5):
    """
    Load per-label thresholds từ file JSON.
    Nếu không có file → dùng threshold mặc định 0.5 cho tất cả.
    """
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r', encoding='utf-8') as f:
            threshold_dict = json.load(f)
        thresholds = np.array([
            threshold_dict.get(label, default)
            for label in VIGO_EMOTIONS
        ], dtype=np.float32)
        print(f"✅ Loaded tuned thresholds từ: {threshold_path}")
    else:
        thresholds = np.full(NUM_VIGO_LABELS, default, dtype=np.float32)
        print(f"⚠️  Không tìm thấy threshold file → dùng threshold={default} cho tất cả")
    return thresholds


def test():
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN    = 128
    BATCH_SIZE = 32

    print(f"🚀 Đánh giá trên: {DEVICE}")

    # ==================== ĐƯỜNG DẪN ====================
    model_path      = os.path.join(project_root, 'checkpoints','Module#1 ver 2.6', 'best_multitask_model.pth')
    threshold_path  = os.path.join(project_root, 'checkpoints', 'Module#1 ver 2.6','emotion_thresholds.json')
    emo_test_path   = os.path.join(project_root, 'data', 'processed', 'emotion_test.csv')
    vihsd_test_path = os.path.join(project_root, 'data', 'processed', 'vihsd_test_clean.xlsx')

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # ==================== LOAD THRESHOLDS ====================
    thresholds = load_thresholds(threshold_path)

    # ==================== LOAD TEST DATA ====================
    df_emo  = pd.read_csv(emo_test_path)
    df_hate = pd.read_excel(vihsd_test_path)

    ds_emo = MultiTaskDataset(
        texts=df_emo['text'].values,
        emotion_labels=df_emo['labels'].apply(parse_labels).tolist(),
        hate_labels=np.full(len(df_emo), -100, dtype=np.int64),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        emotion_source='vigo'
    )
    ds_hate = MultiTaskDataset(
        texts=df_hate['cmt_col'].values,
        emotion_labels=None,
        hate_labels=df_hate['labels'].values.astype(np.int64),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        emotion_source='none'
    )
    print(f"✅ Emotion test: {len(ds_emo)} | Hate test: {len(ds_hate)}")

    ds_all = ConcatDataset([ds_emo, ds_hate])
    loader = DataLoader(ds_all, batch_size=BATCH_SIZE)

    # ==================== LOAD MODEL ====================
    model = PhoBERTMultiTask(num_hate_labels=3, num_emotion_labels=NUM_VIGO_LABELS)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # ==================== PREDICT ====================
    emo_probs_all, emo_true_all = [], []
    hate_preds, hate_true       = [], []

    print("\n📊 Đang dự đoán...")
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion_labels = batch['emotion_labels']
            hate_labels    = batch['hate_labels']
            has_emotion    = batch['has_emotion']

            emo_logits, hate_logits = model(input_ids, attention_mask)

            # Emotion — lưu probs để apply tuned threshold sau
            probs  = torch.sigmoid(emo_logits).cpu().numpy()
            mask_e = has_emotion.bool().numpy()
            if mask_e.any():
                emo_probs_all.append(probs[mask_e])
                emo_true_all.append(emotion_labels[mask_e].numpy())

            # Hate — single-label
            h_preds = torch.max(hate_logits, dim=1)[1].cpu().numpy()
            t_h     = hate_labels.numpy()
            mask_h  = t_h != -100
            hate_preds.extend(h_preds[mask_h])
            hate_true.extend(t_h[mask_h])

    # ==================== EMOTION REPORT ====================
    print("\n" + "="*20 + " EMOTION REPORT (Multi-label 28 nhãn) " + "="*20)

    if emo_probs_all:
        all_probs = np.concatenate(emo_probs_all, axis=0)  # [N, 28]
        all_true  = np.concatenate(emo_true_all,  axis=0)  # [N, 28]

        # Áp dụng threshold mặc định 0.5
        preds_default = (all_probs > 0.5).astype(int)
        f1_micro_def  = f1_score(all_true, preds_default, average='micro', zero_division=0)
        f1_macro_def  = f1_score(all_true, preds_default, average='macro', zero_division=0)

        # Áp dụng tuned threshold per-label
        preds_tuned = np.zeros_like(all_probs, dtype=int)
        for i, thresh in enumerate(thresholds):
            preds_tuned[:, i] = (all_probs[:, i] > thresh).astype(int)
        f1_micro_tuned = f1_score(all_true, preds_tuned, average='micro', zero_division=0)
        f1_macro_tuned = f1_score(all_true, preds_tuned, average='macro', zero_division=0)

        print(f"\n📊 So sánh threshold:")
        print(f"   @threshold=0.5:    F1 micro={f1_micro_def:.4f} | F1 macro={f1_macro_def:.4f}")
        print(f"   @tuned threshold:  F1 micro={f1_micro_tuned:.4f} | F1 macro={f1_macro_tuned:.4f}  "
              f"(+{f1_micro_tuned-f1_micro_def:.4f} micro | +{f1_macro_tuned-f1_macro_def:.4f} macro)")

        # Per-label F1 với tuned threshold — sắp xếp từ thấp đến cao
        f1_per = f1_score(all_true, preds_tuned, average=None, zero_division=0)
        sorted_labels = sorted(zip(VIGO_EMOTIONS, f1_per, thresholds), key=lambda x: x[1])

        print("\n⚠️  Nhãn F1 thấp nhất (bottom 10):")
        for label, f1, thresh in sorted_labels[:10]:
            bar = '█' * int(f1 * 20)
            print(f"  {label:20s} F1={f1:.3f} thresh={thresh:.2f} {bar}")

        print("\n✅ Nhãn F1 cao nhất (top 10):")
        for label, f1, thresh in sorted_labels[-10:]:
            bar = '█' * int(f1 * 20)
            print(f"  {label:20s} F1={f1:.3f} thresh={thresh:.2f} {bar}")

    # ==================== HATE SPEECH REPORT ====================
    print("\n" + "="*20 + " HATE SPEECH REPORT " + "="*20)
    print(classification_report(
        hate_true, hate_preds,
        target_names=['Clean', 'Offensive', 'Hate']
    ))


if __name__ == "__main__":
    test()