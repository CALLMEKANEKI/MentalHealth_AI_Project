import pandas as pd
import os

PROJECT_ROOT = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "VMHQA", "xlsx")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Đọc danh sách topic bệnh
disease_topics_file = os.path.join(PROCESSED_DIR, "disease_topics_filtered.txt")
with open(disease_topics_file, "r", encoding="utf-8") as f:
    disease_topics = set(line.strip() for line in f if line.strip())
print(f"Đã đọc {len(disease_topics)} topic bệnh.")

# Đọc và gộp hai file Excel
file1 = os.path.join(RAW_DIR, "[SHARING]_final_full_10066.xlsx")
file2 = os.path.join(RAW_DIR, "[SHARING]_FinalData_10000_18082024.xlsx")
df1 = pd.read_excel(file1, engine='openpyxl')
df2 = pd.read_excel(file2, engine='openpyxl')
df = pd.concat([df1, df2], ignore_index=True)
print(f"Tổng số dòng gốc: {len(df)}")

# Lọc topic
df['Topic'] = df['Topic'].astype(str).str.strip()
df = df[df['Topic'].isin(disease_topics)].copy()
print(f"Số dòng sau lọc topic: {len(df)}")

def clean_cell(val):
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float)) and val == 0:
        return ""
    if isinstance(val, str) and val.strip() in ("0", "0.0", ""):
        return ""
    return str(val).strip()

# Các cột nguồn
source_cols = ['Paragraph 1', 'Paragraph 2', 'Paragraph 3', 'Fact 1', 'Fact 2', 'Fact 3', "Question"]
for col in source_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_cell)
    else:
        df[col] = ""

# Xử lý cột Question (sẽ được thêm vào mỗi dòng nếu có)
df['Question'] = df['Question'].apply(clean_cell) if 'Question' in df.columns else ""

# Tạo danh sách các dòng mới
new_rows = []
for idx, row in df.iterrows():
    topic = row['Topic']
    question = row['Question']
    for col in source_cols:
        content = row[col]
        if content:  # nếu không rỗng
            # Tạo input_text: có thể chỉ là content hoặc content + "\n" + question
            input_text = content 
            new_rows.append({'input_text': input_text, 'label': topic})

# Tạo DataFrame mới
df_new = pd.DataFrame(new_rows)
print(f"Số dòng sau khi tách: {len(df_new)}")

# Lưu CSV
output_csv = os.path.join(PROCESSED_DIR, "vmhqa_split_rows.csv")
df_new.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"Đã lưu vào {output_csv}")