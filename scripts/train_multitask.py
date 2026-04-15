import os
os.environ["TR_SKIP_TORCH_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Import các module của bạn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

from preprocess import TextCleaner
from dataset import MultiTaskDataset
from models import PhoBERTMultiTask
from class_weights import get_emotion_weights

# ==================== HÀM TIỆN ÍCH ====================
def compute_class_weights_for_sampler(labels):
    """Tính trọng số cho từng mẫu để oversampling lớp hiếm"""
    from collections import Counter
    class_counts = Counter(labels)
    num_samples = len(labels)
    weights = np.zeros(num_samples)
    for i, label in enumerate(labels):
        weights[i] = num_samples / class_counts[label]
    return torch.DoubleTensor(weights)

def evaluate_multitask(model, dataloader, device, return_logits=False):
    model.eval()
    emo_preds, emo_actual = [], []
    hate_preds, hate_actual = [], []
    emo_logits_list = [] if return_logits else None
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            hate_labels = batch['hate_labels'].to(device)

            emo_logits, hate_logits = model(input_ids, attention_mask)
            
            if return_logits:
                emo_logits_list.append(emo_logits.cpu().numpy())
            
            _, e_preds = torch.max(emo_logits, dim=1)
            mask_e = emotion_labels != -100
            emo_preds.extend(e_preds[mask_e].cpu().numpy())
            emo_actual.extend(emotion_labels[mask_e].cpu().numpy())

            _, h_preds = torch.max(hate_logits, dim=1)
            mask_h = hate_labels != -100
            hate_preds.extend(h_preds[mask_h].cpu().numpy())
            hate_actual.extend(hate_labels[mask_h].cpu().numpy())

    acc_emo = np.mean(np.array(emo_preds) == np.array(emo_actual)) if emo_actual else 0
    acc_hate = np.mean(np.array(hate_preds) == np.array(hate_actual)) if hate_actual else 0
    
    if return_logits:
        return acc_emo, acc_hate, np.concatenate(emo_logits_list, axis=0)
    return acc_emo, acc_hate

def plot_confusion_matrix(y_true, y_pred, class_names, epoch, save_dir='confusion_matrices'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(f'Confusion Matrix - Emotion - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'emotion_cm_epoch_{epoch}.png'))
    plt.close()
    print(f"📊 Đã lưu confusion matrix tại {save_dir}/emotion_cm_epoch_{epoch}.png")

# ==================== HÀM UNCERTAINTY WEIGHTING ====================
class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss với trọng số tự học dựa trên uncertainty.
    Công thức: L = (1/(2*sigma1^2))*L1 + (1/(2*sigma2^2))*L2 + log(sigma1) + log(sigma2)
    """
    def __init__(self, num_tasks=2, device='cuda'):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
        
    def forward(self, losses):
        # losses: tensor shape [num_tasks]
        precision = torch.exp(-self.log_vars)
        weighted_loss = torch.sum(precision * losses) + torch.sum(self.log_vars)
        return weighted_loss

# ==================== HÀM TRAIN CHÍNH ====================
def train():
    # --- THAM SỐ ---
    EPOCHS = 20               # Tăng lên 20, early stopping sẽ dừng sớm
    BATCH_SIZE = 12
    GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch = 24
    MAX_LEN = 128
    LEARNING_RATE_BACKBONE = 2e-5
    LEARNING_RATE_HEAD = 1e-4
    LABEL_SMOOTHING = 0.05    # Giảm từ 0.1 xuống
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARLY_STOPPING_PATIENCE = 3
    USE_AMP = False            # Mixed precision
    
    print(f"🚀 Device: {DEVICE}")
    if USE_AMP and DEVICE.type == 'cuda':
        print("✅ Mixed Precision (fp16) enabled")
    
    # --- ĐƯỜNG DẪN ---
    uit_path = os.path.join(project_root, 'data', 'processed', 'uit_train_clean.xlsx')
    valid_uit_path = os.path.join(project_root, 'data', 'processed', 'uit_valid_clean.xlsx')
    vihsd_path = os.path.join(project_root, 'data', 'processed', 'vihsd_train_clean.xlsx')
    valid_vihsd_path = os.path.join(project_root, 'data', 'processed', 'vihsd_valid_clean.xlsx')
    save_path = os.path.join(project_root, 'checkpoints', 'best_multitask_model.pth')
    best_emotion_path = os.path.join(project_root, 'checkpoints', 'best_emotion_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # --- 1. LOAD DỮ LIỆU EMOTION (UIT) ---
    df_uit = pd.read_excel(uit_path)
    le_emotion = LabelEncoder()
    df_uit['emotion_label'] = le_emotion.fit_transform(df_uit['Emotion'])
    print(f"📊 Emotion classes: {dict(zip(le_emotion.classes_, range(len(le_emotion.classes_))))}")
    
    data_uit = pd.DataFrame({
        'text': df_uit['Sentence'],
        'emotion': df_uit['emotion_label'],
        'hate': -100
    })
    
    # --- 2. LOAD DỮ LIỆU HATE (ViHSD) ---
    df_vihsd = pd.read_excel(vihsd_path)
    data_vihsd = pd.DataFrame({
        'text': df_vihsd['cmt_col'],
        'emotion': -100,
        'hate': df_vihsd['labels']
    })
    
    # --- 3. GỘP TRAIN ---
    df_combined = pd.concat([data_uit, data_vihsd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Lấy nhãn emotion thực để tính class weights cho loss
    true_emotion_labels = df_combined[df_combined['emotion'] != -100]['emotion'].tolist()
    emotion_loss_weights = get_emotion_weights(true_emotion_labels).to(DEVICE)
    print(f"⚖️ Emotion loss weights: {emotion_loss_weights.cpu().numpy()}")
    
    # --- 4. WEIGHTED SAMPLER cho Emotion (oversampling lớp hiếm) ---
    # Chỉ lấy các mẫu có emotion != -100 để tạo sampler
    emotion_mask = df_combined['emotion'] != -100
    emotion_indices = df_combined[emotion_mask].index.tolist()
    emotion_labels_for_sampler = df_combined.loc[emotion_indices, 'emotion'].tolist()
    
    if len(emotion_labels_for_sampler) > 0:
        sample_weights = compute_class_weights_for_sampler(emotion_labels_for_sampler)
        # Tạo sampler: chỉ áp dụng cho các mẫu emotion, còn mẫu hate (emotion=-100) sẽ không được oversample
        # Nhưng vì chúng ta muốn oversample cả batch, cần tạo sampler cho toàn bộ dataset
        # Cách đơn giản: tạo sampler cho tất cả, nhưng gán weight=1 cho mẫu hate
        full_weights = np.ones(len(df_combined))
        full_weights[emotion_indices] = sample_weights.numpy()
        sampler = WeightedRandomSampler(full_weights, num_samples=len(full_weights), replacement=True)
        shuffle = False  # sampler đã xáo trộn
        print(f"✅ WeightedRandomSampler được khởi tạo với {len(emotion_indices)} mẫu emotion, {len(df_combined)-len(emotion_indices)} mẫu hate")
    else:
        sampler = None
        shuffle = True
    
    # --- 5. VALIDATION DATA ---
    df_uit_val = pd.read_excel(valid_uit_path)
    df_uit_val['emotion_label'] = le_emotion.transform(df_uit_val['Emotion'])
    data_uit_val = pd.DataFrame({
        'text': df_uit_val['Sentence'],
        'emotion': df_uit_val['emotion_label'],
        'hate': -100
    })
    
    df_vihsd_val = pd.read_excel(valid_vihsd_path)
    data_vihsd_val = pd.DataFrame({
        'text': df_vihsd_val['cmt_col'],
        'emotion': -100,
        'hate': df_vihsd_val['labels']
    })
    
    df_val_combined = pd.concat([data_uit_val, data_vihsd_val], ignore_index=True)
    df_val_combined = df_val_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # --- 6. DATALOADER ---
    train_ds = MultiTaskDataset(
        texts=df_combined['text'].values,
        emotion_labels=df_combined['emotion'].values,
        hate_labels=df_combined['hate'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=0  # Để tránh lỗi multiprocessing trên Windows
    )
    
    valid_ds = MultiTaskDataset(
        texts=df_val_combined['text'].values,
        emotion_labels=df_val_combined['emotion'].values,
        hate_labels=df_val_combined['hate'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- 7. MÔ HÌNH ---
    model = PhoBERTMultiTask(
        num_emotion_labels=len(le_emotion.classes_),
        num_hate_labels=3
    )
    model.to(DEVICE)
    
    # --- 8. OPTIMIZER (Differential LR) ---
    optimizer_grouped_parameters = [
        {'params': model.phobert.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.emotion_head.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.hate_head.parameters(), 'lr': LEARNING_RATE_HEAD}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',          # muốn tối đa emotion accuracy
        factor=0.5,          # giảm LR đi một nửa
        patience=2,          # chờ 2 epoch không cải thiện thì giảm
        verbose=True,        # in ra màn hình khi giảm LR
        min_lr=1e-6          # LR tối thiểu
    )

    # --- 9. LOSS FUNCTIONS ---
    criterion_emotion = nn.CrossEntropyLoss(
        weight=emotion_loss_weights,
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING
    )
    criterion_hate = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING
    )
    
    # Uncertainty weighting
    uncertainty_loss = UncertaintyWeightedLoss(num_tasks=2, device=DEVICE)
    
    # # Scheduler: linear warmup + decay
    # total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    # num_warmup_steps = int(0.1 * total_steps)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=num_warmup_steps, 
    #     num_training_steps=total_steps
    # )
    
    # --- 10. MIXED PRECISION SCALER ---
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # --- 11. VÒNG HUẤN LUYỆN ---
    best_avg_acc = 0
    best_emotion_acc = 0
    patience_counter = 0
    emotion_class_names = list(le_emotion.classes_)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"✨ Epoch {epoch}/{EPOCHS}")
        
        for batch_idx, batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion_labels = batch['emotion_labels'].to(DEVICE)
            hate_labels = batch['hate_labels'].to(DEVICE)
            
            # Forward with mixed precision
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                emo_logits, hate_logits = model(input_ids, attention_mask)
                loss_e = criterion_emotion(emo_logits, emotion_labels)
                loss_h = criterion_hate(hate_logits, hate_labels)
                # Uncertainty weighting
                loss = uncertainty_loss(torch.stack([loss_e, loss_h]))
            
            # Backward with scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", E=f"{loss_e.item():.2f}", H=f"{loss_h.item():.2f}")
        
        # --- ĐÁNH GIÁ ---
        val_acc_emo, val_acc_hate = evaluate_multitask(model, valid_loader, DEVICE)
        avg_acc = (val_acc_emo + val_acc_hate) / 2
        print(f"\n📊 Epoch {epoch}: [Emotion Acc: {val_acc_emo:.4f}] | [Hate Acc: {val_acc_hate:.4f}] | Avg: {avg_acc:.4f}")
        
        # In log_vars của uncertainty (để biết trọng số tự học)
        log_vars = uncertainty_loss.log_vars.detach().cpu().numpy()
        weights = np.exp(-log_vars)
        print(f"   Uncertainty weights - Emotion: {weights[0]:.3f}, Hate: {weights[1]:.3f}")

        # ReduceLROnPlateau step dựa trên emotion accuracy
        scheduler.step(val_acc_emo)
        # In LR hiện tại
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Current LR: {current_lr:.2e}")
        
        # --- CONFUSION MATRIX cho Emotion (mỗi 3 epoch) ---
        if epoch % 3 == 0:
            _, _, emo_logits_all = evaluate_multitask(model, valid_loader, DEVICE, return_logits=True)
            # Lấy nhãn thật cho emotion từ validation loader (chỉ emotion)
            y_true = []
            for batch in valid_loader:
                labels = batch['emotion_labels'].numpy()
                mask = labels != -100
                y_true.extend(labels[mask])
            y_pred = np.argmax(emo_logits_all, axis=1)
            if len(y_true) == len(y_pred):
                plot_confusion_matrix(y_true, y_pred, emotion_class_names, epoch)
        
        # --- LƯU BEST MODEL (dựa trên Emotion Acc) ---
        if val_acc_emo > best_emotion_acc:
            best_emotion_acc = val_acc_emo
            torch.save(model.state_dict(), best_emotion_path)
            print(f"⭐ Lưu best emotion model (acc={best_emotion_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Lưu best avg acc (giữ nguyên cách cũ)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Lưu best avg model (avg={best_avg_acc:.4f})")
        
        # --- EARLY STOPPING ---
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping tại epoch {epoch} do emotion acc không cải thiện sau {EARLY_STOPPING_PATIENCE} epochs")
            break
    
    print(f"\n✅ Hoàn thành training. Best emotion acc: {best_emotion_acc:.4f}, Best avg acc: {best_avg_acc:.4f}")
    print(f"📁 Checkpoints lưu tại:\n   - {save_path}\n   - {best_emotion_path}")
    
    # --- VẼ CONFUSION MATRIX CUỐI CÙNG (nếu chưa vẽ) ---
    print("\n📊 Vẽ confusion matrix cho best emotion model...")
    model.load_state_dict(torch.load(best_emotion_path, map_location=DEVICE))
    _, _, emo_logits_all = evaluate_multitask(model, valid_loader, DEVICE, return_logits=True)
    y_true = []
    for batch in valid_loader:
        labels = batch['emotion_labels'].numpy()
        mask = labels != -100
        y_true.extend(labels[mask])
    y_pred = np.argmax(emo_logits_all, axis=1)
    plot_confusion_matrix(y_true, y_pred, emotion_class_names, epoch='final')
    
    return model, le_emotion

if __name__ == "__main__":
    model, label_encoder = train()