import os
os.environ["TR_SKIP_TORCH_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 1. Cấu trúc đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

from preprocess import TextCleaner
from dataset import MultiTaskDataset # Đảm bảo bạn đã cập nhật class này trong dataset.py
from models import PhoBERTMultiTask  # Sử dụng class model 2 đầu
def evaluate_multitask(model, dataloader, device):
    model.eval()
    
    # Khởi tạo các biến tích lũy
    emo_preds, emo_actual = [], []
    hate_preds, hate_actual = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            hate_labels = batch['hate_labels'].to(device)

            emo_logits, hate_logits = model(input_ids, attention_mask)

            # Xử lý dự đoán Emotion
            _, e_preds = torch.max(emo_logits, dim=1)
            # Chỉ lấy những câu có nhãn thực (khác -100)
            mask_e = emotion_labels != -100
            emo_preds.extend(e_preds[mask_e].cpu().numpy())
            emo_actual.extend(emotion_labels[mask_e].cpu().numpy())

            # Xử lý dự đoán Hate Speech
            _, h_preds = torch.max(hate_logits, dim=1)
            mask_h = hate_labels != -100
            hate_preds.extend(h_preds[mask_h].cpu().numpy())
            hate_actual.extend(hate_labels[mask_h].cpu().numpy())

    # Tính Accuracy riêng cho từng task
    acc_emo = np.mean(np.array(emo_preds) == np.array(emo_actual)) if emo_actual else 0
    acc_hate = np.mean(np.array(hate_preds) == np.array(hate_actual)) if hate_actual else 0
    
    return acc_emo, acc_hate

def train():
    # --- THAM SỐ ---
    EPOCHS = 10
    BATCH_SIZE = 16
    MAX_LEN = 128
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ĐƯỜNG DẪN ---
    uit_path = os.path.join(project_root, 'data', 'processed', 'uit_train_clean.xlsx')
    valid_uit_path = os.path.join(project_root, 'data', 'processed', 'uit_valid_clean.xlsx')
    vihsd_path = os.path.join(project_root, 'data', 'processed', 'vihsd_train_clean.xlsx') 
    valid_vihsd_path = os.path.join(project_root, 'data', 'processed', 'vihsd_valid_clean.xlsx')
    save_path = os.path.join(project_root, 'checkpoints', 'best_multitask_model.pth')

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    # --- 2.1 XỬ LÝ DỮ LIỆU ĐA NHIỆM ---
    # Load UIT (Emotion)
    df_uit = pd.read_excel(uit_path)
    le_emotion = LabelEncoder()
    df_uit['emotion_label'] = le_emotion.fit_transform(df_uit['Emotion'])
    
    # Chuẩn bị bảng UIT: text, emotion, hate (gán -100)
    data_uit = pd.DataFrame({
        'text': df_uit['Sentence'],
        'emotion': df_uit['emotion_label'],
        'hate': -100
    })

    # Load ViHSD (Hate Speech)
    df_vihsd = pd.read_excel(vihsd_path)
    # Chuẩn bị bảng ViHSD: text, emotion (gán -100), hate
    data_vihsd = pd.DataFrame({
        'text': df_vihsd['cmt_col'],
        'emotion': -100,
        'hate': df_vihsd['labels']
    })

    # GỘP 2 BỘ DỮ LIỆU
    df_combined = pd.concat([data_uit, data_vihsd], ignore_index=True)
    df_combined = df_combined.sample(frac=1).reset_index(drop=True) # Xáo trộn

    # --- 2.2 XỬ LÝ DỮ LIỆU VALID ĐA NHIỆM ---
    # Load UIT Valid
    df_uit_val = pd.read_excel(valid_uit_path)
    df_uit_val['emotion_label'] = le_emotion.transform(df_uit_val['Emotion'])
    data_uit_val = pd.DataFrame({
        'text': df_uit_val['Sentence'],
        'emotion': df_uit_val['emotion_label'],
        'hate': -100
    })

    # Load ViHSD Valid
    df_vihsd_val = pd.read_excel(valid_vihsd_path)
    data_vihsd_val = pd.DataFrame({
        'text': df_vihsd_val['cmt_col'],
        'emotion': -100,
        'hate': df_vihsd_val['labels']
    })

    # Gộp tập Valid
    df_val_combined = pd.concat([data_uit_val, data_vihsd_val], ignore_index=True)
    df_val_combined = df_val_combined.sample(frac=1).reset_index(drop=True) # Xáo trộn

    # Tạo Valid Loader
    valid_ds = MultiTaskDataset(
        texts=df_val_combined['text'].values,
        emotion_labels=df_val_combined['emotion'].values,
        hate_labels=df_val_combined['hate'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    # --- 3. DATALOADER ---
    train_ds = MultiTaskDataset(
        texts=df_combined['text'].values,
        emotion_labels=df_combined['emotion'].values,
        hate_labels=df_combined['hate'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. KHỞI TẠO MÔ HÌNH ---
    model = PhoBERTMultiTask(num_emotion_labels=len(le_emotion.classes_), num_hate_labels=3)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: "Học nhanh lúc đầu, chậm lúc sau"
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    criterion_emotion = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_hate = nn.CrossEntropyLoss(ignore_index=-100)

    # --- 5. VÒNG LẶP HUẤN LUYỆN ---
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"✨ Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            emotion_labels = batch['emotion_labels'].to(DEVICE)
            hate_labels = batch['hate_labels'].to(DEVICE)

            emo_logits, hate_logits = model(input_ids, attention_mask)
            
            loss_e = criterion_emotion(emo_logits, emotion_labels)
            loss_h = criterion_hate(hate_logits, hate_labels)
            
            batch_loss = loss_e*0.6  + loss_h*0.4  # Điều chỉnh trọng số nếu muốn ưu tiên task nào hơn
            batch_loss.backward()
            
            optimizer.step()
            scheduler.step() # Cập nhật Learning Rate

            total_loss += batch_loss.item()
            loop.set_postfix(loss=f"{batch_loss.item():.4f}", E=f"{loss_e.item():.2f}", H=f"{loss_h.item():.2f}")

        # Cuối mỗi Epoch trong vòng lặp Train
        val_acc_emo, val_acc_hate = evaluate_multitask(model, valid_loader, DEVICE)
        print(f"📊 Accuracy - [Emotion: {val_acc_emo:.4f}] | [Hate: {val_acc_hate:.4f}]")

        # Lưu model dựa trên trung bình cộng Accuracy của 2 Task
        avg_acc = (val_acc_emo + val_acc_hate) / 2
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Đã lưu Model xuất sắc nhất với Avg Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    train()