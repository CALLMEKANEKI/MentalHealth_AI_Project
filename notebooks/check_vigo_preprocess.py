import pandas as pd
import os
import sys

project_root = r"C:\Users\ASPIRE A715 - 42G\Downloads\MentalHealth_AI_Project"
sys.path.insert(0, os.path.join(project_root, 'src'))

from module1.preprocess import TextCleaner

dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
cleaner = TextCleaner(dict_path)

# Load file gốc ViGoEmotions
df_raw = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'train.csv'))

print("=== SO SÁNH TRƯỚC/SAU CLEAN ĐÚNG ===")
diff_count = 0
for i in range(len(df_raw)):
    raw   = str(df_raw['text'].iloc[i])
    clean = cleaner.clean(raw)
    if raw.lower().strip() != clean.lower().strip():
        print(f"\nTRƯỚC: {raw}")
        print(f"SAU:   {clean}")
        diff_count += 1
    if diff_count >= 30:  # chỉ xem 30 ví dụ
        break

# Tìm từ viết tắt tiếng Việt phổ biến chưa được dịch
from collections import Counter
import re

potential_abbrev = []
for text in df_raw['text']:
    tokens = str(text).lower().split()
    for t in tokens:
        # Chỉ lấy token thuần chữ cái, 2-5 ký tự
        t_clean = re.sub(r'[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]', '', t)
        if 2 <= len(t_clean) <= 4:
            potential_abbrev.append(t_clean)

counter = Counter(potential_abbrev)
print("\n=== TOKEN 2-4 KÝ TỰ XUẤT HIỆN NHIỀU ===")
print("(ứng viên từ viết tắt chưa được dịch)")
for token, count in counter.most_common(60):
    print(f"  {token:8s}: {count}")