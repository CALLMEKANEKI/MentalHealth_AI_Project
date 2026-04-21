import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.module2.module2_dataset import MentalHealthMultiLabelDataset
from src.module2.module2_model import MentalHealthMultiLabelClassifier

def top_k_accuracy(labels, probs, k=5):
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]
    correct = 0
    for i in range(len(labels)):
        true_labels = np.where(labels[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        if np.any(np.isin(true_labels, top_k_preds[i])):
            correct += 1
    return correct / len(labels)

def predict_top_k(model, tokenizer, text, label_names, device='cuda', max_len=128, k=5):
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    top_indices = np.argsort(-probs)[:k]
    return [(label_names[i], float(probs[i])) for i in top_indices]

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/best_mental_health_multilabel.pth"
    label_path = "checkpoints/label_names.pkl"
    test_csv = "data/processed/test.csv"
    max_len = 128
    batch_size = 32
    k_values = [1, 5, 10, 20]

    # Load label names
    with open(label_path, 'rb') as f:
        label_names = pickle.load(f)
    num_labels = len(label_names)
    print(f"Loaded {num_labels} labels")

    # Load test data
    test_df = pd.read_csv(test_csv)
    mlb = MultiLabelBinarizer(classes=label_names)
    test_labels = mlb.fit_transform(test_df['label'].apply(lambda x: [x]))
    print(f"Test samples: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    test_dataset = MentalHealthMultiLabelDataset(test_df['text'].values, test_labels, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MentalHealthMultiLabelClassifier(num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dự đoán toàn bộ test set
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # F1 với ngưỡng 0.5
    preds = (all_probs > 0.5).astype(int)
    f1_micro = f1_score(all_labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, preds, average='macro', zero_division=0)
    print(f"\n✅ Test F1 micro: {f1_micro:.4f}, macro: {f1_macro:.4f}")

    # Top‑k accuracy
    for k in k_values:
        acc = top_k_accuracy(all_labels, all_probs, k=k)
        print(f"   Top-{k} accuracy: {acc:.4f}")

    # Dự đoán mẫu (tùy chọn)
    sample_texts = [
        "Tôi cảm thấy buồn bã, mất ngủ, không muốn gặp ai, nghĩ đến cái chết.",
        "Tôi sợ đi thang máy, mỗi lần thấy nhện là hoảng hốt, tay run."
    ]
    print("\n📝 Dự đoán mẫu (top 5):")
    for text in sample_texts:
        top5 = predict_top_k(model, tokenizer, text, label_names, device, k=5)
        print(f"\nText: {text[:80]}...")
        print(f"Top 5: {top5}")

if __name__ == "__main__":
    test()