import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ==================== CẤU HÌNH ====================
input_file = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\data\processed\Dataset-Collect.xlsx"
output_dir = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\data\processed"
os.makedirs(output_dir, exist_ok=True)

# ==================== ĐỌC VÀ LÀM SẠCH NHÃN ====================
df = pd.read_excel(input_file, engine='openpyxl')
text_col = 'text'
label_col = 'label'
df = df[[text_col, label_col]].dropna()
df.columns = ['text', 'label']

def normalize_label(s):
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

df['label'] = df['label'].apply(normalize_label)
df = df[df['label'] != ""]

# Lưu lại file đã làm sạch
df.to_excel(input_file, index=False, engine='openpyxl')
print(f"✅ Đã làm sạch và lưu file: {input_file}")

# ==================== GỘP CÁC LỚP HIẾM ====================
label_counts = df['label'].value_counts()
# Gộp các lớp có số lượng < min_samples vào "Khác"
min_samples = 30
rare_labels = label_counts[label_counts < min_samples].index.tolist()
if rare_labels:
    print(f"⚠️ Gộp {len(rare_labels)} lớp hiếm (số mẫu < {min_samples}) vào 'Khác'")
    df.loc[df['label'].isin(rare_labels), 'label'] = 'Khác'
    # Cập nhật lại label_counts
    label_counts = df['label'].value_counts()

print(f"\nTổng số mẫu: {len(df)}")
print("Phân bố nhãn:")
print(f"Số lượng nhãn duy nhất sau khi gộp lớp hiếm: {len(df['label'].unique())}")

# ==================== CHIA TẬP ====================
X = df['text']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train)} mẫu")
print(f"Val: {len(X_val)} mẫu")
print(f"Test: {len(X_test)} mẫu")

# ==================== LƯU FILE CSV ====================
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')

print("\n✅ Đã lưu thành công:")
print(f"   - {output_dir}/train.csv")
print(f"   - {output_dir}/val.csv")
print(f"   - {output_dir}/test.csv")