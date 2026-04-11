import os
import sys
# Chỉ đường cho Python tìm thấy folder 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

from src.dataset import MentalHealthDataset
from src.models import PhoBERTForSentiment

# 1. Xác định gốc dự án
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_path = os.path.join(project_root, 'data', 'raw', 'UIT-VSEC', 'test_nor_811.xlsx')
model_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

class SimpleCleaner:
    def clean(self, text):
        # Đây là nơi xử lý văn bản, trả về chuỗi đã làm sạch
        return str(text).lower().strip()

my_cleaner = SimpleCleaner()

def evaluate():
    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy file test tại: {test_path}")
        return

    print(f"🚀 Đang đánh giá mô hình trên thiết bị: {device}")
    
    # Load Tokenizer và Data
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    df_test = pd.read_excel(test_path)
    
    label_map = {label: i for i, label in enumerate(['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise'])}
    
    # Kiểm tra xem tên cột trong file Excel có đúng là 'Emotion' và 'Sentence' không
    test_labels = df_test['Emotion'].map(label_map).values
    
    test_ds = MentalHealthDataset(
            df_test['Sentence'].values, 
            test_labels, 
            tokenizer, 
            cleaner=my_cleaner, 
            max_len=128
        )    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Load Model
    model = PhoBERTForSentiment(num_labels=7)
    # Dùng model_path đã định nghĩa phía trên
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("📊 Đang chạy dự đoán trên tập Test...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán chỉ số
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print("\n" + "="*40)
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print(f"📈 Cohen's Kappa: {kappa:.4f}")
    print("="*40)
    print("\nChi tiết từng nhãn cảm xúc (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))

if __name__ == "__main__":
    evaluate()