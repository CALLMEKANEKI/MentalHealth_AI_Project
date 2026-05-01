import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.module2.module2_dataset import MentalHealthMultiLabelDataset
from src.module2.module2_model import MentalHealthMultiLabelClassifier

def evaluate(model, dataloader, device, k_list=[1,5,10,20]):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = (all_probs > 0.5).astype(int)
    f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
    # Top-k accuracy
    topk_acc = {}
    for k in k_list:
        top_k_preds = np.argsort(-all_probs, axis=1)[:, :k]
        correct = 0
        for i in range(len(all_labels)):
            true = np.where(all_labels[i] == 1)[0]
            if len(true) == 0:
                continue
            if np.any(np.isin(true, top_k_preds[i])):
                correct += 1
        topk_acc[f'top{k}'] = correct / len(all_labels)
    return f1_micro, f1_macro, topk_acc

def train():
    # ==================== THAM SỐ ====================
    EPOCHS = 20
    BATCH_SIZE = 32
    MAX_LEN = 128
    LR_BACKBONE = 2e-5
    LR_HEAD = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOP_PATIENCE = 5
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 0.01

    print(f"🚀 Device: {DEVICE}")

    # ==================== ĐƯỜNG DẪN ====================
    data_dir = "data/processed"
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    save_path = "checkpoints/best_mental_health_multilabel.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # ==================== ĐỌC DỮ LIỆU ====================
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    all_labels = sorted(train_df['label'].unique())
    num_labels = len(all_labels)
    print(f"Số lượng nhãn: {num_labels}")
    print("Ví dụ 10 nhãn đầu:", all_labels[:10])

    mlb = MultiLabelBinarizer(classes=all_labels)
    train_labels = mlb.fit_transform(train_df['label'].apply(lambda x: [x]))
    val_labels = mlb.transform(val_df['label'].apply(lambda x: [x]))
    test_labels = mlb.transform(test_df['label'].apply(lambda x: [x]))

    # Tính pos_weight cho cân bằng
    pos_freq = train_labels.sum(axis=0)
    neg_freq = len(train_labels) - pos_freq

    # Giới hạn pos_weight tối đa để tránh nhãn hiếm được weight quá cao
    raw_weight = neg_freq / (pos_freq + 1e-6)
    MAX_WEIGHT = 20.0  # Không cho weight vượt quá 20x
    pos_weight = torch.tensor(
        np.clip(raw_weight, 1.0, MAX_WEIGHT), 
        dtype=torch.float
    ).to(DEVICE)

    print(f"pos_weight: min={pos_weight.min():.1f}, max={pos_weight.max():.1f}, mean={pos_weight.mean():.1f}")

    # ==================== TOKENIZER & DATALOADER ====================
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    train_dataset = MentalHealthMultiLabelDataset(train_df['text'].values, train_labels, tokenizer, MAX_LEN)
    val_dataset = MentalHealthMultiLabelDataset(val_df['text'].values, val_labels, tokenizer, MAX_LEN)
    # test_dataset không dùng trong train, chỉ để tham khảo

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ==================== MÔ HÌNH ====================
    model = MentalHealthMultiLabelClassifier(num_labels=num_labels).to(DEVICE)

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer_grouped = [
        {'params': model.phobert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ]
    optimizer = AdamW(optimizer_grouped, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ==================== HUẤN LUYỆN ====================
    best_f1 = 0.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_f1_micro, val_f1_macro, val_topk = evaluate(model, val_loader, DEVICE)
        avg_loss = total_loss / len(train_loader)
        print(f"📊 Epoch {epoch}: Loss={avg_loss:.4f}, Val F1 micro={val_f1_micro:.4f}, macro={val_f1_macro:.4f}")
        print(f"   Top-1: {val_topk['top1']:.4f}, Top-5: {val_topk['top5']:.4f}, Top-10: {val_topk['top10']:.4f}")

        if val_f1_micro > best_f1:
            best_f1 = val_f1_micro
            torch.save(model.state_dict(), save_path)
            # Lưu label names
            with open(os.path.join("checkpoints", "label_names.pkl"), "wb") as f:
                pickle.dump(all_labels, f)
            print(f"⭐ Saved best model (F1 micro={best_f1:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print("Training completed.")

if __name__ == "__main__":
    train()