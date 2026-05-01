import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from collections import Counter

# ==================== CẤU HÌNH ====================
input_file = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\data\processed\Dataset-Collect.xlsx"
output_dir = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project\data\processed"
os.makedirs(output_dir, exist_ok=True)

MIN_SAMPLES = 30        # Ngưỡng gộp vào nhóm
MAX_KHAC_RATIO = 0.10   # "Khác" không được vượt quá 10% dataset
ABSOLUTE_MIN = 10       # Dưới ngưỡng này → loại bỏ hẳn, không gộp

# ==================== ĐỌC VÀ LÀM SẠCH ====================
df = pd.read_excel(input_file, engine='openpyxl')
df = df[['text', 'label']].dropna()
df.columns = ['text', 'label']

def normalize_label(s):
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

df['label'] = df['label'].apply(normalize_label)
df = df[df['label'] != ""]

label_counts = df['label'].value_counts()
print(f"Tổng nhãn ban đầu: {len(label_counts)}")
print(f"Tổng mẫu ban đầu: {len(df)}")

# ==================== PHÂN LOẠI NHÃN HIẾM ====================
# Nhãn có < ABSOLUTE_MIN dòng → loại bỏ hẳn (quá ít để học)
remove_labels = label_counts[label_counts < ABSOLUTE_MIN].index.tolist()

# Nhãn có ABSOLUTE_MIN <= count < MIN_SAMPLES → gộp vào "Khác"  
rare_labels = label_counts[
    (label_counts >= ABSOLUTE_MIN) & (label_counts < MIN_SAMPLES)
].index.tolist()

print(f"\n⛔ Loại bỏ {len(remove_labels)} nhãn có < {ABSOLUTE_MIN} mẫu")
print(f"⚠️  Gộp {len(rare_labels)} nhãn có {ABSOLUTE_MIN}-{MIN_SAMPLES} mẫu vào 'Khác'")

# Loại bỏ nhãn quá hiếm
df = df[~df['label'].isin(remove_labels)]

# Gộp nhãn hiếm vào "Khác"
df.loc[df['label'].isin(rare_labels), 'label'] = 'Khác'

# ==================== KIỂM TRA TỶ LỆ "Khác" ====================
label_counts = df['label'].value_counts()
khac_ratio = label_counts.get('Khác', 0) / len(df)
print(f"\n📊 'Khác' hiện chiếm: {khac_ratio*100:.1f}%")

# Nếu "Khác" vẫn > MAX_KHAC_RATIO → undersample "Khác"
if khac_ratio > MAX_KHAC_RATIO:
    target_khac = int(len(df) * MAX_KHAC_RATIO)
    khac_df = df[df['label'] == 'Khác'].sample(n=target_khac, random_state=42)
    other_df = df[df['label'] != 'Khác']
    df = pd.concat([other_df, khac_df]).reset_index(drop=True)
    print(f"✂️  Undersample 'Khác' xuống còn {target_khac} dòng ({MAX_KHAC_RATIO*100:.0f}%)")

# ==================== THỐNG KÊ CUỐI ====================
label_counts = df['label'].value_counts()
print(f"\n✅ Tổng mẫu sau xử lý: {len(df)}")
print(f"✅ Số nhãn duy nhất: {len(label_counts)}")
print(f"✅ 'Khác' chiếm: {label_counts.get('Khác',0)/len(df)*100:.1f}%")
print("\nTop 20 nhãn:")
print(label_counts.head(20).to_string())

# ==================== CHIA TẬP ====================
X, y = df['text'], df['label']

# Kiểm tra nhãn nào có quá ít mẫu để stratify
min_count = y.value_counts().min()
if min_count < 2:
    print(f"\n⚠️ Có nhãn chỉ có {min_count} mẫu, không thể stratify!")
    stratify = None
else:
    stratify = y

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42,
    stratify=y_temp if stratify is not None else None
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==================== LƯU FILE ====================
pd.DataFrame({'text': X_train, 'label': y_train}).to_csv(
    os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'text': X_val, 'label': y_val}).to_csv(
    os.path.join(output_dir, 'val.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(
    os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')

print("\n✅ Đã lưu train/val/test.csv thành công!")

# ==================== BÁO CÁO CUỐI ====================
print("\n=== BÁO CÁO PHÂN BỐ CUỐI ===")
for label, count in label_counts.items():
    bar = '█' * (count // 20)
    print(f"{label[:40]:40s} {count:4d} {bar}")