import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
import pickle
from sklearn.metrics import classification_report

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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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

    
    # Thêm vào test.py sau phần in Top-k accuracy

    print("\n=== PER-LABEL F1 REPORT ===")
    preds_binary = (all_probs > 0.5).astype(int)

    # Lấy index của label "Khác"
    khac_idx = label_names.index('Khác') if 'Khác' in label_names else None

    report = classification_report(
        all_labels, 
        preds_binary,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    # In các nhãn có F1 thấp nhất
    print("\n⚠️  Nhãn có F1 thấp nhất (bottom 10):")
    sorted_labels = sorted(
        [(k, v['f1-score']) for k, v in report.items() 
        if k not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']],
        key=lambda x: x[1]
    )
    for label, f1 in sorted_labels[:10]:
        print(f"  {label[:45]:45s} F1={f1:.3f}")

    print("\n✅ Nhãn có F1 cao nhất (top 10):")
    for label, f1 in sorted_labels[-10:]:
        print(f"  {label[:45]:45s} F1={f1:.3f}")

    if khac_idx is not None:
        khac_f1 = report.get('Khác', {}).get('f1-score', 0)
        khac_precision = report.get('Khác', {}).get('precision', 0)
        khac_recall = report.get('Khác', {}).get('recall', 0)
        print(f"\n🔍 'Khác' riêng: Precision={khac_precision:.3f} | Recall={khac_recall:.3f} | F1={khac_f1:.3f}")

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

    
    # # Thêm vào test.py để xem confusion matrix của 2 nhãn này
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # # Tìm index của 2 nhãn
    # idx_tl = label_names.index('Rối loạn tâm lý')
    # idx_tt = label_names.index('Rối loạn tâm thần')

    # # Xem model đang nhầm chúng với nhãn nào
    # print("\n=== PHÂN TÍCH NHÃN CONFUSE ===")
    # for idx, name in [(idx_tl, 'Rối loạn tâm lý'), (idx_tt, 'Rối loạn tâm thần')]:
    #     true_pos = all_labels[:, idx]
    #     pred_pos = (all_probs[:, idx] > 0.5).astype(int)
        
    #     # Khi nhãn này đúng là 1, model predict gì thay vào?
    #     wrong_samples = np.where((true_pos == 1) & (pred_pos == 0))[0]
    #     print(f"\n'{name}' bị miss {len(wrong_samples)} samples")
    #     print("Model thay vào đó predict:")
    #     for i in wrong_samples[:5]:  # xem 5 mẫu đầu
    #         top_pred_idx = np.argsort(-all_probs[i])[:3]
    #         top_preds = [(label_names[j], f"{all_probs[i][j]:.2f}") for j in top_pred_idx]
    #         print(f"  → {top_preds}")


if __name__ == "__main__":
    test()