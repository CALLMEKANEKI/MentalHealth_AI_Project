import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src', "module1"))

from module1.dataset import MultiTaskDataset
from module1.models import PhoBERTMultiTask
from module1.preprocess import TextCleaner

# --- CẤU HÌNH ---
model_path = os.path.join(project_root, 'checkpoints', 'best_multitask_model.pth')
uit_test_path = os.path.join(project_root, 'data', 'processed','uit_test_clean.xlsx')
vihsd_test_path = os.path.join(project_root, 'data', 'processed', 'vihsd_test_clean.xlsx') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"🚀 Đang đánh giá Multi-task Model trên: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    cleaner = TextCleaner(os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx'))

    # 1. Load Data (Tương tự như lúc Train nhưng dành cho Test)
    df_uit = pd.read_excel(uit_test_path)
    df_vihsd = pd.read_excel(vihsd_test_path)

    # Mapping nhãn Emotion (Phải khớp với lúc Train)
    emotion_list = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']
    emo_map = {label: i for i, label in enumerate(emotion_list)}

    # Gộp data test để chạy 1 lượt (hoặc chạy riêng từng bộ tùy Giang)
    data_test = pd.concat([
        pd.DataFrame({'text': df_uit['Sentence'], 'emo': df_uit['Emotion'].map(emo_map), 'hate': -100}),
        pd.DataFrame({'text': df_vihsd['cmt_col'], 'emo': -100, 'hate': df_vihsd['labels']})
    ], ignore_index=True)

    test_ds = MultiTaskDataset(
        texts=data_test['text'].values,
        emotion_labels=data_test['emo'].values,
        hate_labels=data_test['hate'].values,
        tokenizer=tokenizer,
        max_len=128  
    )
    test_loader = DataLoader(test_ds, batch_size=16)

    # 2. Load Model Multi-task
    model = PhoBERTMultiTask(num_emotion_labels=7, num_hate_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    emo_preds, emo_true = [], []
    hate_preds, hate_true = [], []

    print("📊 Đang dự đoán...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Model trả về 2 output
            logits_e, logits_h = model(input_ids, attention_mask)
            
            # Lấy dự đoán
            p_e = torch.max(logits_e, dim=1)[1].cpu().numpy()
            p_h = torch.max(logits_h, dim=1)[1].cpu().numpy()
            
            # Lấy nhãn thật
            t_e = batch['emotion_labels'].numpy()
            t_h = batch['hate_labels'].numpy()

            # Chỉ lưu những câu có nhãn thực sự (khác -100)
            mask_e = t_e != -100
            emo_preds.extend(p_e[mask_e])
            emo_true.extend(t_e[mask_e])

            mask_h = t_h != -100
            hate_preds.extend(p_h[mask_h])
            hate_true.extend(t_h[mask_h])

    # 3. Hiển thị kết quả
    print("\n" + "="*20 + " EMOTION REPORT " + "="*20)
    print(classification_report(emo_true, emo_preds, target_names=emotion_list))

    print("\n" + "="*20 + " HATE SPEECH REPORT " + "="*20)
    print(classification_report(hate_true, hate_preds, target_names=['Clean', 'Offensive', 'Hate']))

if __name__ == "__main__":
    evaluate()