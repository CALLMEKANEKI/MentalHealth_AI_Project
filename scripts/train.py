import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import sys
import torch
torch.serialization.add_safe_globals(['set'])
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim import AdamW

# Thiết lập đường dẫn để import từ thư mục src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

from preprocess import TextCleaner
from dataset import MentalHealthDataset
from models import PhoBERTForSentiment

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
    return correct_predictions.double() / len(dataloader.dataset), np.mean(losses)

def train():
    # --- 1. CẤU HÌNH THAM SỐ  ---
    EPOCHS = 5
    BATCH_SIZE = 16
    MAX_LEN = 128
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {DEVICE}")

    # --- 2. ĐƯỜNG DẪN FILE ---
    dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
    train_path = os.path.join(project_root, 'data', 'raw', 'UIT-VSEC', 'train_nor_811.xlsx')
    valid_path = os.path.join(project_root, 'data', 'raw', 'UIT-VSEC', 'valid_nor_811.xlsx')
    save_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')

    # Tạo thư mục checkpoints nếu chưa có
    os.makedirs(os.path.join(project_root, 'checkpoints'), exist_ok=True)

    # --- 3. KHỞI TẠO CÔNG CỤ ---
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    cleaner = TextCleaner(dict_path)
    
    # Đọc dữ liệu Excel
    df_train = pd.read_excel(train_path)
    df_valid = pd.read_excel(valid_path)

    # Mã hóa nhãn Emotion thành số
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(df_train['Emotion'])
    valid_labels = label_encoder.transform(df_valid['Emotion'])
    num_labels = len(label_encoder.classes_)
    
    print(f"📍 Các cảm xúc nhận diện: {label_encoder.classes_}")

    # --- 4. TẠO DATALOADER ---
    train_ds = MentalHealthDataset(df_train['Sentence'].values, train_labels, tokenizer, cleaner, MAX_LEN)
    valid_ds = MentalHealthDataset(df_valid['Sentence'].values, valid_labels, tokenizer, cleaner, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    # --- 5. KHỞI TẠO MÔ HÌNH ---
    model = PhoBERTForSentiment(num_labels=num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0

    # --- 6. VÒNG LẶP HUẤN LUYỆN ---
    for epoch in range(EPOCHS):
        print(f"\n✨ Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        train_losses = []
        
        loop = tqdm(train_loader, desc="🔥 Training")
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            loop.set_description(f"Loss: {loss.item():.4f}")

        # Đánh giá sau mỗi Epoch
        val_acc, val_loss = evaluate(model, valid_loader, criterion, DEVICE)
        print(f"📊 Kết quả: Train Loss: {np.mean(train_losses):.4f} | Val Acc: {val_acc:.4f}")

        # Lưu model tốt nhất
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            print(f"⭐ Đã lưu Model mới tốt nhất tại: {save_path}")

if __name__ == "__main__":
    train()