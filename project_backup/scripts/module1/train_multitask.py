# =============================================================================
# Module 1 - Train Multitask (Ver 2.5)
# =============================================================================
# Thay đổi so với các ver trước:
#   - Ver 2.2: Baseline tốt nhất (F1 micro=0.5287), không Sampler
#   - Ver 2.3: Thêm Sampler → làm GIẢM performance (-0.013)
#   - Ver 2.4: Giữ Sampler + fix preprocess → vẫn thua 2.2
#   - Ver 2.5: Bỏ Sampler (quay về shuffle=True như 2.2)
#              + Giữ preprocess fix (HTML, zero-width) từ 2.4
#              + Giữ teencode mới đã thêm
#              + pos_weight tính 1 lần trước khi train (không dynamic)
#              Kỳ vọng: F1 micro ~0.53-0.54
# =============================================================================

import os
os.environ["TR_SKIP_TORCH_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import ast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

current_dir  = os.path.dirname(os.path.abspath(__file__))
scripts_dir  = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

print(f"📁 project_root: {project_root}")

from module1.dataset import MultiTaskDataset, NUM_VIGO_LABELS
from module1.model import PhoBERTMultiTask


def parse_labels(label_str):
    try:
        return ast.literal_eval(str(label_str))
    except Exception:
        return [27]


def compute_pos_weight(df_emo, num_labels=28):
    """
    Tính pos_weight từ training data.
    Gọi 1 LẦN DUY NHẤT trước khi train — không tính lại mỗi epoch
    vì df_emo không thay đổi trong quá trình train.
    """
    all_labels = np.zeros(num_labels)
    for label_str in df_emo['labels']:
        indices = ast.literal_eval(str(label_str))
        for idx in indices:
            if 0 <= idx < num_labels:
                all_labels[idx] += 1
    total      = len(df_emo)
    neg_freq   = total - all_labels
    pos_weight = neg_freq / (all_labels + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)
    return torch.tensor(pos_weight, dtype=torch.float)


def evaluate_multitask(model, dataloader, device, threshold=0.5):
    model.eval()
    emo_preds_all, emo_true_all = [], []
    hate_preds, hate_actual     = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            hate_labels    = batch['hate_labels'].to(device)
            has_emotion    = batch['has_emotion'].to(device)

            emo_logits, hate_logits = model(input_ids, attention_mask)

            emo_pred_bin = (torch.sigmoid(emo_logits) > threshold).float()
            mask_e = has_emotion.bool()
            if mask_e.any():
                emo_preds_all.append(emo_pred_bin[mask_e].cpu().numpy())
                emo_true_all.append(emotion_labels[mask_e].cpu().numpy())

            _, h_preds = torch.max(hate_logits, dim=1)
            mask_h = hate_labels != -100
            hate_preds.extend(h_preds[mask_h].cpu().numpy())
            hate_actual.extend(hate_labels[mask_h].cpu().numpy())

    if emo_preds_all:
        emo_preds_np = np.concatenate(emo_preds_all, axis=0)
        emo_true_np  = np.concatenate(emo_true_all,  axis=0)
        f1_micro = f1_score(emo_true_np, emo_preds_np, average='micro', zero_division=0)
        f1_macro = f1_score(emo_true_np, emo_preds_np, average='macro', zero_division=0)
    else:
        f1_micro = f1_macro = 0.0

    acc_hate = np.mean(np.array(hate_preds) == np.array(hate_actual)) if hate_actual else 0.0
    return f1_micro, f1_macro, acc_hate


def train():
    # ==================== THAM SỐ ====================
    EPOCHS              = 30
    BATCH_SIZE          = 12    # giữ nguyên như ver 2.2
    MAX_LEN             = 128
    DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOP_PATIENCE = 7
    THRESHOLD           = 0.5

    print(f"🚀 Device: {DEVICE}")
    print(f"   Ver 2.5: No Sampler (shuffle=True) + preprocess fix")

    # ==================== ĐƯỜNG DẪN ====================
    # Emotion: file đã merge từ prepare_data.py (có preprocess fix ver 2.4)
    emo_train_path   = os.path.join(project_root, 'data', 'processed', 'emotion_train.csv')
    emo_valid_path   = os.path.join(project_root, 'data', 'processed', 'emotion_valid.csv')

    # Hate Speech
    vihsd_train_path = os.path.join(project_root, 'data', 'processed', 'vihsd_train_clean.xlsx')
    vihsd_valid_path = os.path.join(project_root, 'data', 'processed', 'vihsd_valid_clean.xlsx')

    save_path = os.path.join(project_root, 'checkpoints', 'best_multitask_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # ==================== LOAD DATA ====================
    df_emo_train  = pd.read_csv(emo_train_path)
    df_emo_valid  = pd.read_csv(emo_valid_path)
    df_hate_train = pd.read_excel(vihsd_train_path)
    df_hate_valid = pd.read_excel(vihsd_valid_path)

    print(f"✅ Emotion  train={len(df_emo_train)} | valid={len(df_emo_valid)}")
    print(f"✅ Hate     train={len(df_hate_train)} | valid={len(df_hate_valid)}")

    # ==================== TẠO DATASET ====================
    def make_ds(df_emo, df_hate, tokenizer, max_len):
        from torch.utils.data import ConcatDataset
        ds_emo = MultiTaskDataset(
            texts=df_emo['text'].values,
            emotion_labels=df_emo['labels'].apply(parse_labels).tolist(),
            hate_labels=np.full(len(df_emo), -100, dtype=np.int64),
            tokenizer=tokenizer,
            max_len=max_len,
            emotion_source='vigo'
        )
        ds_hate = MultiTaskDataset(
            texts=df_hate['cmt_col'].values,
            emotion_labels=None,
            hate_labels=df_hate['labels'].values.astype(np.int64),
            tokenizer=tokenizer,
            max_len=max_len,
            emotion_source='none'
        )
        return ConcatDataset([ds_emo, ds_hate])

    train_ds = make_ds(df_emo_train, df_hate_train, tokenizer, MAX_LEN)
    val_ds   = make_ds(df_emo_valid, df_hate_valid, tokenizer, MAX_LEN)

    # ==================== DATALOADER ====================
    # Ver 2.5: Dùng shuffle=True — KHÔNG dùng WeightedRandomSampler
    # Lý do: Sampler (ver 2.3, 2.4) làm giảm F1 hate từ 0.81 → 0.78
    # shuffle=True (ver 2.2) cho kết quả tốt hơn
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2)

    print(f"\n📊 Train total: {len(train_ds)} | Val total: {len(val_ds)}")
    print(f"   Loader: shuffle=True (NO WeightedRandomSampler)")

    # ==================== MÔ HÌNH ====================
    model = PhoBERTMultiTask(num_hate_labels=3, num_emotion_labels=NUM_VIGO_LABELS)
    model.to(DEVICE)

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = AdamW([
        {'params': model.phobert.parameters(),      'lr': 2e-5},
        {'params': model.emotion_head.parameters(), 'lr': 1e-4},
        {'params': model.hate_head.parameters(),    'lr': 1e-4}
    ])
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ==================== LOSS ====================
    # pos_weight tính 1 lần duy nhất — df_emo_train không đổi trong train
    pos_weight = compute_pos_weight(df_emo_train).to(DEVICE)
    print(f"   pos_weight — min={pos_weight.min():.1f} | max={pos_weight.max():.1f} | mean={pos_weight.mean():.1f}")

    criterion_emotion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    criterion_hate    = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1, reduction='none')

    # ==================== TRAINING LOOP ====================
    best_score = 0.0
    patience   = 0
    nan_count_total = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        nan_count  = 0

        # Trọng số loss theo giai đoạn
        if epoch + 1 <= 3:
            w_e, w_h = 0.5, 0.5
            strategy = "Warm-up: Equal Focus"
        elif epoch + 1 <= 15:
            w_e, w_h = 0.7, 0.3
            strategy = "Main Training: Emotion Focus"
        else:
            w_e, w_h = 0.8, 0.2
            strategy = "Final Squeeze: Max Emotion Focus"

        print(f"\n--- Chiến lược Epoch {epoch+1}: {strategy} [E:{w_e} - H:{w_h}] ---")
        loop = tqdm(train_loader, desc=f"✨ Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion_labels = batch['emotion_labels'].to(DEVICE)
            hate_labels    = batch['hate_labels'].to(DEVICE)
            has_emotion    = batch['has_emotion'].to(DEVICE)

            emo_logits, hate_logits = model(input_ids, attention_mask)

            losses     = []
            loss_e_val = loss_h_val = 0.0

            # ── Emotion loss ──────────────────────────────────────────
            mask_e = has_emotion.bool()
            if mask_e.any():
                loss_e_each = criterion_emotion(
                    emo_logits[mask_e], emotion_labels[mask_e]
                ).mean(dim=1)
                loss_e = loss_e_each.mean()
                if not torch.isnan(loss_e):
                    losses.append(loss_e * w_e)
                    loss_e_val = loss_e.item()

            # ── Hate loss ─────────────────────────────────────────────
            mask_h = hate_labels != -100
            if mask_h.any():
                loss_h_each = criterion_hate(
                    hate_logits[mask_h], hate_labels[mask_h]
                )
                loss_h = loss_h_each.mean()
                if not torch.isnan(loss_h):
                    losses.append(loss_h * w_h)
                    loss_h_val = loss_h.item()

            # ── Skip batch nếu không có loss hợp lệ ──────────────────
            if not losses:
                nan_count += 1
                continue

            batch_loss = sum(losses)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item()
            loop.set_postfix(
                loss=f"{batch_loss.item():.4f}",
                E=f"{loss_e_val:.2f}",
                H=f"{loss_h_val:.2f}"
            )

        # ── Đánh giá sau mỗi epoch ───────────────────────────────────
        f1_micro, f1_macro, acc_hate = evaluate_multitask(
            model, val_loader, DEVICE, THRESHOLD
        )
        avg_loss = total_loss / max(len(train_loader) - nan_count, 1)
        nan_count_total += nan_count

        print(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f} | NaN batches skipped={nan_count}")
        print(f"   Emotion → F1 micro={f1_micro:.4f} | F1 macro={f1_macro:.4f}")
        print(f"   Hate    → Acc={acc_hate:.4f}")

        # Combined score: ưu tiên emotion vì khó hơn
        combined_score = (f1_micro * 0.6) + (acc_hate * 0.4)
        print(f"   Combined score: {combined_score:.4f}")

        if combined_score > best_score:
            best_score = combined_score
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Đã lưu best model (score={best_score:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping tại Epoch {epoch+1}")
                break

    print(f"\n✅ Training hoàn tất.")
    print(f"   Best combined score: {best_score:.4f}")
    print(f"   Tổng NaN batches skipped: {nan_count_total}")


if __name__ == "__main__":
    train()