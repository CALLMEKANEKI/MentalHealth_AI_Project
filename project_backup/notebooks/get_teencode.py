import pandas as pd
import re
from collections import Counter
import os

def find_new_teencode(uit_path, vihsd_path, dict_path, output_path):
    # 1. Load từ điển hiện tại của Giang để biết cái gì đã có rồi
    df_existing = pd.read_excel(dict_path)
    existing_teencodes = set(df_existing['Teencode - GenZ'].astype(str).str.lower().str.strip().values)

    # 2. Load 2 bộ dataset
    df_uit = pd.read_excel(uit_path)
    df_vihsd = pd.read_excel(vihsd_path)

    # Gộp tất cả văn bản lại (Đảm bảo loại bỏ giá trị trống và ép kiểu string)
    text_uit = df_uit['Sentence'].dropna().astype(str).tolist()
    text_vihsd = df_vihsd['cmt_col'].dropna().astype(str).tolist()
    
    all_text = " ".join(text_uit) + " " + " ".join(text_vihsd)    
    # 3. Tách từ (bao gồm cả các ký tự đặc biệt để bắt icon như :v, :3)
    # Regex này bắt các từ hoặc các cụm ký tự đặc biệt đứng liền nhau
    words = re.findall(r'\w+|[^\w\s]+', all_text.lower())

    # 4. Đếm tần suất xuất hiện
    word_counts = Counter(words)

    # 5. Lọc ra những từ/icon chưa có trong từ điển và không phải là từ thuần Việt quá phổ biến
    # Mình sẽ lấy những từ có độ dài ngắn (thường là teencode) hoặc chứa ký tự đặc biệt (icon)
    new_items = []
    for word, count in word_counts.items():
        if word not in existing_teencodes and len(word) > 0:
            # Điều kiện: là ký tự đặc biệt (icon) HOẶC từ ngắn (<= 5 ký tự) xuất hiện nhiều
            if not word.isalnum() or (len(word) <= 5 and count > 5):
                new_items.append({'Teencode - GenZ': word, 'Tần suất': count})

    # 6. Sắp xếp theo tần suất xuất hiện nhiều nhất và lưu ra Excel
    df_new = pd.DataFrame(new_items).sort_values(by='Tần suất', ascending=False)
    
    # Chỉ lấy top 500 cái phổ biến nhất 
    df_new.to_excel(output_path, index=False)
    print(f"✅ Đã quét xong! Giang mở file '{output_path}' để xem nhé.")

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
dict_path = os.path.join(project_root, 'data', 'external', 'Xử lý teencode.xlsx')
uit_path = os.path.join(project_root, 'data', 'raw', 'UIT-VSEC', 'train_nor_811.xlsx')
vihsd_path = os.path.join(project_root, 'data', 'raw', 'ViHSD', 'xlsx', 'train_df.xlsx')
output_path = os.path.join(project_root, 'data', 'external', 'potential_teencode.xlsx')

if __name__ == "__main__":
    find_new_teencode(uit_path, vihsd_path, dict_path, output_path)