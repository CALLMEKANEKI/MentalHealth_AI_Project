"""
export_vigo_tokens.py
─────────────────────
Export toàn bộ token độc nhất trong ViGoEmotions (sau khi clean)
ra file txt để review thủ công và thêm vào từ điển teencode.

Output: data/external/vigo_tokens_review.txt
Format:
    token | count | (để trống → điền nghĩa nếu là từ viết tắt)
"""

import os
import sys
import re
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"✅ Added project root to sys.path: {project_root}")
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from module1.preprocess import TextCleaner

# ==================== CẤU HÌNH ====================
VIGO_TRAIN = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'train.csv')
VIGO_VALID = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'valid.csv')
VIGO_TEST  = os.path.join(project_root, 'data', 'raw', 'ViGoEmotions', 'test.csv')
DICT_PATH  = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
OUTPUT_TXT = os.path.join(project_root, 'data', 'external', 'vigo_tokens_review.txt')

# ==================== LOAD CLEANER & TEENCODE ====================
cleaner = TextCleaner(DICT_PATH)

# Lấy tất cả từ đã có trong từ điển (để đánh dấu "đã dịch")
df_dict = pd.read_excel(DICT_PATH)
existing_teencode = set(
    str(k).lower().strip()
    for k in df_dict['Teencode - GenZ']
    if pd.notna(k)
)
print(f"✅ Từ điển hiện có: {len(existing_teencode)} từ")

# ==================== LOAD & CLEAN DATA ====================
dfs = []
for path in [VIGO_TRAIN, VIGO_VALID, VIGO_TEST]:
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
        print(f"✅ Loaded: {os.path.basename(path)} ({len(dfs[-1])} rows)")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Tổng: {len(df_all)} rows")

# Clean toàn bộ text
print("🧹 Đang clean text...")
df_all['text_clean'] = df_all['text'].apply(
    lambda x: cleaner.clean(str(x))
)

# ==================== TRÍCH XUẤT NGUYÊN TỪ (WHOLE WORDS) ====================
print("🔍 Đang trích xuất toàn bộ từ vựng...")

word_counter = Counter()

for text in df_all['text']:
    # 1. Chuyển về chữ thường
    text = str(text).lower()
    
    # 2. Tách từ theo khoảng trắng (Giữ nguyên cấu trúc từ)
    # Chúng ta không dùng tokenizer của transformers ở đây
    raw_words = text.split()
    
    for word in raw_words:
        # 3. Làm sạch dấu câu ở đầu/cuối từ (ví dụ: "vcl!!!" -> "vcl")
        # Nhưng không xóa dấu câu ở giữa (ví dụ: "v.c.l" vẫn giữ "v.c.l")
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        
        if clean_word:
            word_counter[clean_word] += 1

# Sắp xếp theo số lần xuất hiện
sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

# ==================== EXPORT KẾT QUẢ ====================
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write(f"{'TỪ (WORD)':<25} | {'SỐ LẦN':<8} | {'TRẠNG THÁI':<20} | NGHĨA\n")
    f.write("-" * 80 + "\n")

    for word, count in sorted_words:
        # Kiểm tra xem từ này đã có trong từ điển teencode chưa
        # Lưu ý: existing_teencode cần được load từ file Excel của bạn
        if word in existing_teencode:
            status = "✅ Đã có"
        else:
            status = "❓ Chưa có"
        
        f.write(f"{word:<25} | {count:<8} | {status:<20} |\n")

print(f"✅ Đã lưu {len(sorted_words)} từ vào file: {OUTPUT_TXT}")

import os
import re
import pandas as pd
from collections import Counter

# ==================== ĐƯỜNG DẪN ====================
# (Giữ nguyên các path cũ của bạn)
DICT_PATH  = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
OUTPUT_TXT = os.path.join(project_root, 'data', 'external', 'vigo_tokens_review.txt')
VN_DICT_FILE = os.path.join(project_root, 'data', 'external', 'Viet74K.txt')

# ==================== LOAD TỪ ĐIỂN TIẾNG VIỆT CHUẨN ====================
standard_words = set()
if os.path.exists(VN_DICT_FILE):
    with open(VN_DICT_FILE, 'r', encoding='utf-8') as f:
        # File này mỗi dòng là một từ, có thể là từ ghép (sinh viên)
        # Chúng ta tách ra để lấy tập hợp các tiếng đơn chuẩn
        for line in f:
            parts = line.strip().lower().split()
            standard_words.update(parts)
    print(f"✅ Đã nạp {len(standard_words)} từ đơn tiếng Việt chuẩn.")
else:
    print("⚠️ Không tìm thấy file 'vietnamese_words.txt'. Script sẽ liệt kê TOÀN BỘ từ (không lọc).")

# ==================== LOAD TEENCODE ĐÃ CÓ ====================
df_dict = pd.read_excel(DICT_PATH)
existing_teencode = set(str(k).lower().strip() for k in df_dict['Teencode - GenZ'] if pd.notna(k))
print(f"✅ Từ điển teencode hiện có: {len(existing_teencode)} từ")

# ==================== TRÍCH XUẤT TOÀN BỘ TỪ (WHOLE WORDS) ====================
print("🔍 Đang gom nhóm từ vựng từ ViGo...")
word_counter = Counter()

for text in df_all['text']:
    # Tách từ theo khoảng trắng (Giữ nguyên từ, không chẻ nhỏ)
    tokens = str(text).lower().split()
    for t in tokens:
        # Làm sạch dấu câu ở 2 đầu (giữ lại dấu giữa từ như v.c.l)
        clean_w = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', t)
        if clean_w:
            word_counter[clean_w] += 1

# Sắp xếp theo tần suất xuất hiện
sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)

# ==================== EXPORT KẾT QUẢ ====================
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write("DANH SÁCH REVIEW TỪ VỰNG VIGO\n")
    f.write(f"{'TỪ (WORD)':<25} | {'SỐ LẦN':<8} | {'TRẠNG THÁI'}\n")
    f.write("-" * 70 + "\n")

    new_teencode_count = 0
    
    for word, count in sorted_words:
        # Phân loại
        if word in existing_teencode:
            status = "✅ Đã có trong Dict"
        elif word in standard_words:
            # Nếu là từ tiếng Việt chuẩn, ta bỏ qua (không in ra file) để đỡ rối
            continue 
        else:
            # Đây là những từ lạ (teencode, viết tắt, hoặc sai chính tả)
            status = "❓ CẦN REVIEW"
            new_teencode_count += 1
            
        f.write(f"{word:<25} | {count:<8} | {status}\n")

print(f"✅ Đã xuất {new_teencode_count} từ 'lạ' cần review ra: {OUTPUT_TXT}")